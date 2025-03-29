import datetime
import os
import random
import copy
from argparse import ArgumentParser
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT

# from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassConfusionMatrix
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch.distributed as dist

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from lightning_fabric.strategies import FSDPStrategy
from backbones.encoder_bert import BertEncoder
from backbones.encoder_pemed import DinoEncoder, CausalLMEncoder
from backbones.loss import NTXentLoss
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from backbones.heads import SimCLRProjectionHead
from huggingface_hub import PyTorchModelHubMixin
import torch._dynamo

# from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam


from dataset.utils import get_specificity_with_sensitivity, pfbeta

torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHEXPERT_BASE_CAPTION = "this is a chest x ray of a patient with "


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

os.environ["WANDB_START_METHOD"] = "thread"


class MaMACLIP(
    LightningModule,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        img_encoder: str = "dinov2_vitb14_reg",
        freeze_llm: bool = False,
        emb_dim: int = 128,
        softmax_temperature: float = 0.07,
        learning_rate: float = 2e-5,
        momentum: float = 0.9,
        weight_decay: float = 0.05,
        batch_size: int = 144,
        num_workers: int = 8,
        num_heads: int = 1,
        lamb: float = 0.75,
        epsilon: float = 0.05,
        peft: str = None,
        agg_tokens: bool = False,
        grad_ckpt: bool = False,
        img_cls_ft: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        if 'config' in self.hparams:
            self.hparams.update(self.hparams.config)

        if self.hparams.embed:
            self.hparams.num_classes = 4 if self.hparams.pred_density else 7
        elif self.hparams.rsna_mammo:
            self.hparams.num_classes = 2
        else:
            self.hparams.num_classes = self.hparams.num_classes

        self.confmat = MulticlassConfusionMatrix(self.hparams.num_classes)
        self.all_scores = None
        self.all_labels = None

        # init encoders
        self.img_encoder_q = DinoEncoder(
            model_name=img_encoder,
            output_dim=self.hparams.emb_dim,
            linear_proj=True,
            freeze_vit=self.hparams.freeze_vit,
            pretrained=(not self.hparams.random_vit),
            vit_grad_ckpt=self.hparams.vit_grad_ckpt,
            img_size=self.hparams.crop_size,
        )

        # Randomize the visual transformer
        if self.hparams.random_vit:
            self.img_encoder_q.model.init_weights()

        # Create a text encoder
        if not self.hparams.img_cls_ft:
            if self.hparams.llm_type == "bert":
                self.text_encoder_q = BertEncoder(
                    output_dim=self.hparams.emb_dim,
                    freeze_llm=self.hparams.freeze_llm,
                    agg_tokens=self.hparams.agg_tokens,
                )
            else:
                self.text_encoder_q = CausalLMEncoder(
                    output_dim=self.hparams.emb_dim,
                    freeze_llm=self.hparams.freeze_llm,
                    peft=self.hparams.peft,
                    agg_tokens=self.hparams.agg_tokens,
                    grad_ckpt=self.hparams.grad_ckpt,
                    llm_type=self.hparams.llm_type,
                    linear_proj=True,
                    unlock_ln=self.hparams.unlock_ln,
                    total_steps=self.hparams.max_steps,
                    num_freeze_blocks=self.hparams.num_freeze_blocks,
                    avg_sent_feat=self.hparams.avg_sent_feat,
                )

        # Load pre-trained vit parameter
        if self.hparams.pretrained_encoder != None:
            print(
                "\n### Loading pretrained model from {}\n".format(
                    self.hparams.pretrained_encoder
                )
            )
            state_dict = torch.load(
                self.hparams.pretrained_encoder, map_location="cpu"
            )["state_dict"]
            img_encoder_state_dict = {
                k.replace("img_encoder_q.", ""): v
                for k, v in state_dict.items()
                if k.startswith("img_encoder_q")
            }
            missing, unexpected = self.img_encoder_q.load_state_dict(
                img_encoder_state_dict, strict=False
            )
            print("### Missing keys: ", missing)
            print("### Unexpected keys: ", unexpected)
            if not self.hparams.img_cls_ft:
                text_encoder_state_dict = {
                    k.replace("text_encoder_q.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("text_encoder_q")
                }
                self.text_encoder_q.load_state_dict(text_encoder_state_dict)

        # create a global classifier
        if self.hparams.img_cls_ft:
            self.img_encoder_q.global_embed = nn.Linear(
                self.img_encoder_q.feature_dim, self.hparams.num_classes
            )
            self.img_encoder_q.global_embed.weight.requires_grad = True
            self.img_encoder_q.global_embed.bias.requires_grad = True

        # Initialize the learnable logit scale
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.hparams.softmax_temperature)
        )
        if self.hparams.local_contrast:
            self.local_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / self.hparams.softmax_temperature)
            )
            # freeze local parameters before late loss
            if self.hparams.late_loss > 0:
                self.local_scale.requires_grad = False
                for param in self.img_encoder_q.local_embed.parameters():
                    param.requires_grad = False
                if not self.hparams.img_cls_ft:
                    for param in self.text_encoder_q.local_embed.parameters():
                        param.requires_grad = False

        self.zero_shot_text_feats = None

        # Create extra slip training components
        if self.hparams.slip:
            self.simclr_proj = SimCLRProjectionHead(
                self.img_encoder_q.feature_dim,
                self.img_encoder_q.feature_dim,
                self.hparams.emb_dim,
            )
            self.simclr_loss = NTXentLoss(gather_distributed=(self.hparams.devices > 1))

        # Freeze unused parameters:
        if self.hparams.pool_feat:
            self.img_encoder_q.model.norm.weight.requires_grad = False
            self.img_encoder_q.model.norm.bias.requires_grad = False
        if not self.hparams.local_contrast:
            self.img_encoder_q.local_embed = nn.Identity()
            if not self.hparams.img_cls_ft:
                self.text_encoder_q.local_embed = nn.Identity()

    def get_data_keys(self, split="train"):
        # 50% of chance to use unpaired text
        # Only provide unpaired text for training
        keys = ["imgs", "caption_ids", "attention_mask", "multi_hot_label"]
        return keys

    # @profile
    def forward(self, batch, batch_idx, split="train"):
        """Forward step of our method"""
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        try:
            report_feat_q_full, word_feat_q_full, word_attn_q_full, sents_feat = (
                self.text_encoder_q(
                    batch[cap_key],
                    batch[attn_key],
                    token_type=batch.get("token_type_ids", None),
                )
            )
        except Exception as e:
            print(batch[cap_key].shape)
            print(batch["path"])
            raise e
        if self.hparams.pool_txt_feat:
            report_feat_q_full = word_feat_q_full.mean(dim=1)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
        report_emb_q = F.normalize(report_emb_q, dim=-1)

        ########### image-text contrastive loss ################
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()
        scores = img_emb_q.mm(report_emb_q.t())
        scores *= self.logit_scale.exp()
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_c = loss0 + loss1

        # following slip, we add SimCLR projection results
        ########### image-image contrastive loss ################
        if self.hparams.slip and self.global_step >= self.hparams.late_loss:
            ext_feat_s1, _, ext_full1 = self.img_encoder_q(batch["ext_imgs"])
            if self.hparams.pool_feat:
                ext_feat_s1 = ext_full1.mean(dim=1)
            ext_feat_s2 = img_feat_q
            ext_emb_s1 = self.simclr_proj(ext_feat_s1)
            ext_emb_s2 = self.simclr_proj(ext_feat_s2)
            simclr_loss = self.simclr_loss(ext_emb_s1, ext_emb_s2)
            loss_c += self.hparams.slip_loss_lambda * simclr_loss

            ########### symmetric clip loss ################
            if self.hparams.symmetric_clip:
                ext_emb_q = self.img_encoder_q.global_embed(ext_feat_s1)
                ext_emb_q = F.normalize(ext_emb_q, dim=-1)
                ext_scores = ext_emb_q.mm(report_emb_q.t())
                ext_scores *= self.logit_scale.exp()
                ext_scores1 = ext_scores.transpose(0, 1)
                ext_loss0 = F.cross_entropy(ext_scores, labels)
                ext_loss1 = F.cross_entropy(ext_scores1, labels)
                loss_c += 1.0 * (ext_loss0 + ext_loss1)

        patch_emb_q = None
        ########### local image-text contrastive loss ################
        if self.hparams.local_contrast and self.global_step >= self.hparams.late_loss:
            t2i_local_scores = []
            i2t_local_scores = []
            bsz = patch_feat_q.size(0)
            labels = torch.arange(bsz).type_as(patch_feat_q).long()
            patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
            patch_emb_q = F.normalize(patch_emb_q, dim=-1)  # N x num_patch x C
            for idx, sent_emb_q in enumerate(sents_feat):  # N
                sent_emb_q = self.text_encoder_q.local_embed(sent_emb_q)
                sent_emb_q = F.normalize(sent_emb_q, dim=-1)  # num_sent x C
                sent_scores = torch.einsum(
                    "npc,sc->nps", patch_emb_q, sent_emb_q.squeeze()
                )
                # Max over space + Avg over sentence
                t2i_sent_scores = sent_scores.max(dim=1)[0].mean(dim=1)
                t2i_local_scores.append(t2i_sent_scores)
                if self.hparams.symmetric_local:
                    # Max over sentence + Avg over space
                    i2t_patch_scores = sent_scores.max(dim=2)[0].mean(dim=1)
                    i2t_local_scores.append(i2t_patch_scores)
            t2i_local_scores = torch.stack(t2i_local_scores, dim=0)
            t2i_local_scores *= self.local_scale.exp()
            loss0 = F.cross_entropy(t2i_local_scores, labels)
            loss1 = F.cross_entropy(t2i_local_scores.t(), labels)
            loss_c += 1.0 * (loss0 + loss1)
            if self.hparams.symmetric_local:
                i2t_local_scores = torch.stack(i2t_local_scores, dim=0)
                i2t_local_scores *= self.local_scale.exp()
                loss0 = F.cross_entropy(i2t_local_scores, labels)
                loss1 = F.cross_entropy(i2t_local_scores.t(), labels)
                loss_c += 1.0 * (loss0 + loss1)

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.0
        acc5 = (i2t_acc5 + t2i_acc5) / 2.0

        return loss_c, acc1, acc5

    def zero_shot_inference(self, batch, batch_idx, split="test"):
        """Inference with zero shot setting"""
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        with torch.no_grad():
            # Forward of query image encoder
            img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
            # Following FLIP, use the average of patch features w/o layer norm
            if self.hparams.pool_feat:
                img_feat_q = img_full.mean(dim=1)
            # Use classification token instead of averaged patch tokens
            img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
            img_emb_q = F.normalize(img_emb_q, dim=-1)

            # Forward of query text encoder
            # Forward for each individual image
            bsz = img_emb_q.size(0)  # N x C
            batch_scores = []
            if batch[cap_key].shape[0] == 1:
                raise ValueError
            if not self.hparams.instance_test_cap:
                fixed_caption_ids = batch[cap_key][0]  # CLS x S, get rid of batch dim
                fixed_attention_mask = batch[attn_key][0]

            for idx in range(bsz):
                if self.hparams.instance_test_cap:
                    fixed_caption_ids = batch[cap_key][idx]
                    fixed_attention_mask = batch[attn_key][idx]
                if self.zero_shot_text_feats is None or self.hparams.instance_test_cap:
                    token_type = batch.get("token_type_ids", None)
                    token_type = None if token_type is None else token_type[idx]
                    (
                        report_feat_q_full,
                        word_feat_q_full,
                        word_attn_q_full,
                        sents_full,
                    ) = self.text_encoder_q(
                        fixed_caption_ids, fixed_attention_mask, token_type=token_type
                    )
                    report_emb_q = self.text_encoder_q.global_embed(report_feat_q_full)
                    report_emb_q = F.normalize(report_emb_q, dim=-1)

                    self.zero_shot_text_feats = report_emb_q  # CLS x C

                scores = img_emb_q[idx : idx + 1].mm(
                    self.zero_shot_text_feats.t()
                )  # 1 x CLS
                scores *= self.logit_scale.exp()
                batch_scores.append(scores.squeeze(0))
            scores = torch.stack(batch_scores, dim=0)  # N x CLS

            ########### image-text zero-shot cls loss ################
            labels = batch[label_key].type_as(scores)  # N x CLS

            # Image to text classification loss
            loss0 = F.cross_entropy(scores, labels.argmax(dim=-1))

            # compute retrieval accuracy
            i2t_acc1 = self.precision_at_k(scores, labels.argmax(dim=-1), top_k=(1,))[0]

            labels = labels.float().detach().cpu().numpy()
            scores = torch.softmax(scores.float().detach(), dim=1).cpu().numpy()
            # auc = roc_auc_score(labels, scores)
            auc = 0.0
            # report = classification_report(np.argmax(labels, axis=-1), np.argmax(scores, axis=-1),
            #                                output_dict=True, zero_division=0)

            if split == "test":
                if self.hparams.devices > 1:
                    score_list = [
                        torch.zeros_like(scores) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(score_list, scores)
                    all_scores = torch.cat(score_list, dim=0)
                    label_list = [
                        torch.zeros_like(labels) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(label_list, labels)
                    all_labels = torch.cat(label_list, dim=0)
                else:
                    all_scores = torch.tensor(scores)
                    all_labels = torch.tensor(labels)
                self.confmat.update(
                    torch.argmax(all_scores, dim=-1), all_labels.argmax(dim=-1)
                )
                all_scores = all_scores.detach().to(torch.float32)
                all_scores = torch.softmax(all_scores, dim=-1).cpu().numpy()
                all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
                if self.all_scores is None:
                    self.all_scores = all_scores
                else:
                    self.all_scores = np.concatenate(
                        [self.all_scores, all_scores], axis=0
                    )
                if self.all_labels is None:
                    self.all_labels = all_labels
                else:
                    self.all_labels = np.concatenate(
                        [self.all_labels, all_labels], axis=0
                    )

        return loss0, i2t_acc1, auc

    def visual_forward(self, batch, batch_idx, split="train"):
        """Forward step of our method"""
        img_key, cap_key, attn_key, label_key = self.get_data_keys(split)

        # Forward of query image encoder
        img_feat_q, patch_feat_q, img_full = self.img_encoder_q(batch[img_key])
        # Following FLIP, use the average of patch features w/o layer norm
        if self.hparams.pool_feat:
            img_feat_q = img_full.mean(dim=1)
        # Use classification token instead of averaged patch tokens
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)

        ########### Classification loss ################
        labels = batch[label_key].type_as(img_emb_q)  # N x CLS

        # Image classification loss
        loss0 = F.cross_entropy(img_emb_q, labels.argmax(dim=-1))

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            img_emb_q, labels.argmax(dim=-1), top_k=(1, 2)
        )

        if split == "test":
            if self.hparams.devices > 1:
                img_emb_q_list = [
                    torch.zeros_like(img_emb_q) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(img_emb_q_list, img_emb_q)
                all_img_emb_qs = torch.cat(img_emb_q_list, dim=0)
                label_list = [
                    torch.zeros_like(labels) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(label_list, labels)
                all_labels = torch.cat(label_list, dim=0)
            else:
                all_img_emb_qs = img_emb_q
                all_labels = labels
            self.confmat.update(
                torch.argmax(all_img_emb_qs, dim=-1), all_labels.argmax(dim=-1)
            )
            all_img_emb_qs = all_img_emb_qs.detach().to(torch.float32)
            all_img_emb_qs = torch.softmax(all_img_emb_qs, dim=-1).cpu().numpy()
            all_labels = all_labels.detach().to(torch.float32).cpu().numpy()
            if self.all_scores is None:
                self.all_scores = all_img_emb_qs
            else:
                self.all_scores = np.concatenate(
                    [self.all_scores, all_img_emb_qs], axis=0
                )
            if self.all_labels is None:
                self.all_labels = all_labels
            else:
                self.all_labels = np.concatenate([self.all_labels, all_labels], axis=0)

        return loss0, i2t_acc1, i2t_acc5

    def training_step(self, batch, batch_idx):
        # unlock params after late loss starting step
        if self.hparams.late_loss > 0 and self.global_step == self.hparams.late_loss:
            if self.hparams.local_contrast:
                self.local_scale.requires_grad = True
                for param in self.img_encoder_q.local_embed.parameters():
                    param.requires_grad = True
                for param in self.text_encoder_q.local_embed.parameters():
                    param.requires_grad = True

        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(batch, batch_idx, "train")
        else:
            loss_c, acc1, acc5 = self(batch, batch_idx, "train")
        loss = loss_c

        log = {
            "train_loss": loss,
            "train_loss_c": loss_c,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.img_cls_ft:
            loss_c, acc1, acc5 = self.visual_forward(batch, batch_idx, "val")
        else:
            loss_c, acc1, acc5 = self(batch, batch_idx, "val")
        loss = loss_c

        log = {
            "val_loss": loss,
            "val_loss_c": loss_c,
            "val_acc1": acc1,
            "val_acc5": acc5,
        }
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def test_step(self, batch, batch_idx):

        if self.hparams.img_cls_ft:
            loss_c, acc1, auc = self.visual_forward(batch, batch_idx, "test")
        else:
            loss_c, acc1, auc = self.zero_shot_inference(batch, batch_idx, "test")
        loss = loss_c

        log = {
            "test_loss": loss,
            "test_loss_c": loss_c,
            "test_acc1": acc1,
            "test_auc": auc,
        }
        self.log_dict(
            log,
            batch_size=self.hparams.batch_size,
            sync_dist=True,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def on_test_epoch_end(self):

        # Calculate the confusion matrix using the accumulated predictions and targets
        conf_matrix = self.confmat.compute().cpu().numpy()
        print("\n\n### Confusion Matrix:\n", conf_matrix)
        if self.hparams.rsna_mammo:
            tn = conf_matrix[0, 0]
            tp = conf_matrix[1, 1]
            fn = conf_matrix[1, 0]
            fp = conf_matrix[0, 1]
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            f1 = 2 * tp / (2 * tp + fp + fn)
            print("\n### Sensitivity: {:.4f}".format(100 * sensitivity))
            print("### Specificity: {:.4f}".format(100 * specificity))
            print("### PPV: {:.4f}".format(100 * ppv))
            print("### NPV: {:.4f}".format(100 * npv))
            print("### F1: {:.4f}".format(100 * f1))
        cls_cnt = np.sum(conf_matrix, axis=1)
        cls_hit = np.diag(conf_matrix)
        cls_acc = cls_hit / cls_cnt
        print("\n### Class Accuracy: ", [f"{100 * acc:.4f}" for acc in cls_acc])
        # Calculate the accuracy using the accumulated predictions and targets
        idx_label = np.argmax(self.all_labels, -1)
        idx_pred = np.argmax(self.all_scores, -1)
        acc = 100 * accuracy_score(idx_label, idx_pred)
        # f1 = 100 * f1_score(idx_label, idx_pred)
        ba = 100 * balanced_accuracy_score(idx_label, idx_pred)
        try:
            if self.hparams.num_classes == 2:
                auc = 100 * roc_auc_score(idx_label, self.all_scores[:, 1])
                spec_80 = 100 * get_specificity_with_sensitivity(
                    idx_label, self.all_scores[:, 1], 0.8
                )
                pF1 = 100 * pfbeta(idx_label, self.all_scores[:, 1])
            else:
                auc = 100 * roc_auc_score(idx_label, self.all_scores, multi_class="ovr")
                spec_80 = 0.0
                pF1 = 0.0
        except Exception as e:
            print("### Warning: AUC calculation failed with error:", e)
            auc = 0
            spec_80 = 0.0
            pF1 = 0.0
        print("### Accuracy: {:.4f}".format(acc))
        print("### Balanced Accuracy: {:.4f}".format(ba))
        print("### AUC: {:.4f}".format(auc))
        print("### pF1: {:.4f}".format(pF1))

        # Reset metrics for the next test run
        self.confmat.reset()
        self.all_scores = None
        self.all_labels = None

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        """Compute the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = (
                    correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                )
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def multi_label_precision(
        output: torch.Tensor, target: torch.Tensor, threshold=0.5
    ):
        """Compute the accuracy over the k top predictions for the specified values"""
        with torch.no_grad():
            # Applying threshold to prediction probabilities
            preds = output > threshold

            # Correct output are only those where prediction and label are equal
            correct_preds = (preds == target).float()

            # Compute accuracy across all target
            accuracy = 100 * correct_preds.sum() / (len(target) * target.size(1))

            return accuracy

    def configure_optimizers(self):
        parameters = self.parameters()
        if self.hparams.sgd:
            optimizer = torch.optim.SGD(
                parameters,
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                parameters,
                self.hparams.learning_rate,
                betas=(self.hparams.momentum, 0.999),
                weight_decay=self.hparams.weight_decay,
            )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.hparams.max_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warm_up,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Model args
        parser.add_argument("--emb_dim", type=int, default=128, help="128, 256, 512")
        parser.add_argument("--pool_feat", action="store_true")
        parser.add_argument("--pool_txt_feat", action="store_true")
        ### Visual Model args
        parser.add_argument("--img_encoder", type=str, default="dinov2_vitb14_reg")
        parser.add_argument("--freeze_vit", action="store_true")
        parser.add_argument("--slip", action="store_true")
        parser.add_argument("--symmetric_clip", action="store_true")
        parser.add_argument("--slip_loss_lambda", type=float, default=1.0)
        parser.add_argument("--random_vit", action="store_true")
        parser.add_argument("--vit_grad_ckpt", action="store_true")
        parser.add_argument("--stochastic_depth_prob", type=float, default=0.0)
        ### LLM args
        parser.add_argument(
            "--llm_type",
            type=str,
            default="gpt",
            help="bert, gpt, llama, llama2, or llama3",
        )
        parser.add_argument("--freeze_llm", action="store_true")
        parser.add_argument("--unlock_ln", action="store_true")
        parser.add_argument("--avg_sent_feat", action="store_true")
        parser.add_argument("--num_freeze_blocks", type=int, default=0)
        parser.add_argument("--peft", type=str, default=None)

        # Training args
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--max_epochs", type=int, default=50)  # Unused
        parser.add_argument("--max_steps", type=int, default=40000)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--img_cls_ft", action="store_true")
        parser.add_argument("--num_classes", type=int, default=1000)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--devices", type=int, default=4)
        parser.add_argument("--strategy", type=str, default="ddp")
        parser.add_argument("--accelerator", type=str, default="gpu")
        parser.add_argument("--precision", type=str, default="32")
        parser.add_argument("--dev", action="store_true")
        parser.add_argument("--grad_ckpt", action="store_true")
        parser.add_argument("--warm_up", type=int, default=16000)
        parser.add_argument("--balance_training", action="store_true")
        parser.add_argument("--balance_ratio", type=int, default=-1)
        parser.add_argument("--local_contrast", action="store_true")
        parser.add_argument("--symmetric_local", action="store_true")
        parser.add_argument("--late_loss", type=int, default=-1)
        ### Hyperparameters
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--min_lr", type=float, default=1e-8)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        ### Optimizer
        parser.add_argument("--sgd", action="store_true")
        ### Pretrained args
        parser.add_argument("--pretrained_encoder", type=str, default=None)
        parser.add_argument("--use_flash_attention", action="store_true")

        # Data args
        parser.add_argument("--embed", action="store_true")
        parser.add_argument("--rsna_mammo", action="store_true")
        parser.add_argument("--agg_tokens", action="store_true")
        parser.add_argument("--data_pct", type=float, default=1.0)
        parser.add_argument("--train_split", type=str, default="train")
        parser.add_argument("--valid_split", type=str, default="valid")
        parser.add_argument("--load_jpg", action="store_true")
        parser.add_argument("--img_size", type=int, default=224)
        parser.add_argument("--crop_size", type=int, default=224)
        ### EMBED test set args
        parser.add_argument("--balanced_test", action="store_true")
        parser.add_argument("--small_balanced_train", action="store_true")
        parser.add_argument("--pred_density", action="store_true")
        # Caption args
        parser.add_argument("--structural_cap", action="store_true")
        parser.add_argument("--simple_cap", action="store_true")
        parser.add_argument("--natural_cap", action="store_true")
        parser.add_argument("--mask_ratio", type=float, default=0.0)
        parser.add_argument("--mask_meta", type=float, default=-1.0)
        # EMBED multi-images args
        parser.add_argument("--inter_view", action="store_true")
        parser.add_argument("--inter_side", action="store_true")
        # Inference args
        parser.add_argument("--instance_test_cap", action="store_true")

        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPStrategy, FSDPStrategy))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""

        return trainer.max_steps
