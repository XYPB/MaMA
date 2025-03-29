import datetime
import os
from argparse import ArgumentParser

# from torch.utils.tensorboard import SummaryWriter

import torch
from dateutil import tz

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from dataset.data_module import DataModule
from dataset.pretrain_dataset import multimodal_collate_fn, EmbedPretrainingDataset
from dataset.mammo_eval_dataset import RSNAMammo
from dataset.transforms import (
    Moco2Transform,
    SimCLRTransform,
)

from model import MaMACLIP

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


os.environ["WANDB_START_METHOD"] = "thread"


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def train(args, model, datamodule):
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension += f"_{args.experiment_name}"
    ckpt_dir = os.path.join(BASE_DIR, f"logs/ckpts/MaMACLIP/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=ckpt_dir,
            save_last=True,
            mode="min",
            save_top_k=1,
        ),
        # EarlyStopping(monitor="val_loss", min_delta=0.,
        #               patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(BASE_DIR, f"./logs")
    os.makedirs(logger_dir, exist_ok=True)
    if args.img_cls_ft:
        if args.embed:
            project = "MaMACLIP_img_embed_ft"
        else:
            project = "MaMACLIP_img_cls_ft"
        if "fft" in args.experiment_name:
            project = project.replace("ft", "fft")
    elif args.embed:
        project = "MaMACLIP_Embed"
    else:
        project = "MaMACLIP_fix_step"
    wandb_logger = WandbLogger(project=project, save_dir=logger_dir, name=extension)
    num_available_gpus = torch.cuda.device_count()
    if args.devices > num_available_gpus:
        print(
            f"### Using less GPUs than requested: {args.devices} > {num_available_gpus}"
        )
        args.devices = num_available_gpus
    print(f"### Using {args.strategy} Strategy with {args.devices} GPUs")
    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        fast_dev_run=args.dev,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=int(1 / args.data_pct),
    )

    model.training_steps = model.num_training_steps(trainer, datamodule)

    dtype = None
    if args.strategy == "fsdp":
        for name, param in model.named_parameters():
            if dtype is None:
                dtype = param.dtype
            elif dtype != param.dtype:
                print(f"Parameter {name} has dtype {param.dtype}, expected {dtype}")
        print(dtype)

    print(f"\n### Resume from {args.resume_ckpt}...\n")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_ckpt,
    )
    trainer.test(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)
    return model


def eval(args, model, datamodule):
    model.eval()

    # Single GPU inference
    trainer = Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        devices=1,
        fast_dev_run=args.dev,
        max_epochs=1,
        deterministic=args.deterministic,
        inference_mode=True,
    )

    trainer.test(model, datamodule=datamodule)


def cli_main():

    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser = MaMACLIP.add_model_specific_args(parser)

    args = parser.parse_args()
    args.deterministic = False

    if args.eval:
        args.batch_size = 32
        args.data_pct = 1.0
        args.max_epoch = 1
        args.accumulate_grad_batches = 1
        args.dev = False
        args.strategy = None
        args.devices = 1
        args.grad_ckpt = False
    num_cores = len(os.sched_getaffinity(0))
    if args.num_workers > num_cores:
        args.num_workers = num_cores
        print("switching to maximum num_workers = ", num_cores)

    if args.use_flash_attention:
        os.environ["XFORMERS_DISABLED"] = "1"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    # speed-up GEMM for Ampere GPUs
    torch.set_float32_matmul_precision("high")

    # seed
    seed_everything(args.seed)

    if args.embed:
        dataset = EmbedPretrainingDataset
    elif args.rsna_mammo:
        dataset = RSNAMammo
    else:
        dataset = EmbedPretrainingDataset

    if args.slip:
        transform_obj = SimCLRTransform
    else:
        transform_obj = Moco2Transform
    # use default collect function for DataLoader
    collate_fn = multimodal_collate_fn

    datamodule = DataModule(
        dataset,
        collate_fn,
        transform_obj,
        args.data_pct,
        args.batch_size,
        args.num_workers,
        llm_type=args.llm_type,
        train_split=args.train_split,
        valid_split=args.valid_split,
        structural_cap=args.structural_cap,
        simple_cap=args.simple_cap,
        natural_cap=args.natural_cap,
        instance_test_cap=args.instance_test_cap,
        inter_side=args.inter_side,
        inter_view=args.inter_view,
        balanced_test=args.balanced_test,
        slip=args.slip,
        balance_training=args.balance_training,
        pred_density=args.pred_density,
        img_size=args.img_size,
        crop_size=args.crop_size,
        load_jpg=args.load_jpg,
        mask_ratio=args.mask_ratio,
        mask_meta=args.mask_meta,
        balance_ratio=args.balance_ratio,
    )

    if args.pretrained_model is None:
        model = MaMACLIP(**args.__dict__)
    else:
        print(f"\n\n##### Loading pretrained model from {args.pretrained_model}\n\n")
        model = MaMACLIP.load_from_checkpoint(
            args.pretrained_model, map_location="cpu", strict=False, **args.__dict__
        )

    if args.eval:
        eval(args, model, datamodule)
    else:
        model = train(args, model, datamodule)


if __name__ == "__main__":
    cli_main()
