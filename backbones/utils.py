import os
import types
import warnings
from typing import Callable
from enum import Enum
from typing import Union
import torch
from transformers import AutoTokenizer, BertTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_API_TOKEN = "<replace-with-your-hf-api-token>"


class Weights(Enum):
    LVD142M = "LVD142M"


def _parse_dinov2_model_name(dino_model_name):
    # dinov2_vitb14_reg_lc
    items = dino_model_name.split("_")
    num_register_tokens = 4 if items[-1] == "reg" else 0
    model_size = items[1][3]
    patch_size = int(items[1][4:])
    if model_size == "s":
        arch_name = "vit_small"
        if patch_size == 14:
            pretrained = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"
            )
        else:
            pretrained = None
    elif model_size == "b":
        arch_name = "vit_base"
        if patch_size == 14:
            pretrained = os.path.expanduser(
                "~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"
            )
        else:
            pretrained = None
    elif model_size == "l":
        arch_name = "vit_large"
        warnings.warn("Using the large model w/o pretraining.")
        pretrained = None
    else:
        arch_name = "vit_giant2"
        warnings.warn("Using the large model w/o pretraining.")
        pretrained = None
    return arch_name, pretrained, num_register_tokens, patch_size


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: str = None,
    weights: Union[Weights, str] = Weights.LVD142M,
    grad_ckpt: bool = False,
    **kwargs,
):
    import backbones.dino_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        grad_ckpt=grad_ckpt,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        state_dict = torch.load(pretrained, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise e
            # warnings.warn(f"Error loading pretrained weights: {e}")
            # warnings.warn('Init pretrained model w/ interpolated pos_embed')
            # print(state_dict.keys())
            # state_dict.pop('pos_embed')
            # model.load_state_dict(state_dict, strict=False)
    return model


def get_tokenizer(llm_type):
    if llm_type == "gpt":
        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM")
        tokenizer.add_special_tokens(
            {
                "bos_token": "<|startoftext|>",
                "pad_token": "<|padtext|>",
                "mask_token": "<|masktext|>",
                "sep_token": "<|separatetext|>",
                "unk_token": "<|unknowntext|>",
                "additional_special_tokens": [
                    "<|keytext|>",
                ],
            }
        )
    elif llm_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "epfl-llm/meditron-7b", token=MY_API_TOKEN, padding_side="right"
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<KEY>",
                ],
            }
        )
    elif llm_type == "llama2":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", token=MY_API_TOKEN, padding_side="right"
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                "sep_token": "<sep>",
                "additional_special_tokens": [
                    "<key>",
                ],
            }
        )
    elif llm_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", token=MY_API_TOKEN
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|pad_text|>",
                "mask_token": "<|mask_text|>",
                "sep_token": "<|separate_of_text|>",
                "additional_special_tokens": [
                    "<|keyword_of_text|>",
                ],
            }
        )
    else:
        tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        tokenizer.add_special_tokens(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "additional_special_tokens": [
                    "[KEY]",
                ],
            }
        )
    return tokenizer


def random_masking(x, mask_ratio=0.50):
    N, S, D = x.shape

    mask = torch.rand(N, S, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(mask, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, : int(S * (1 - mask_ratio))]

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    return x_masked, ids_restore


def masked_only_prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)
    if masks is not None:
        x = torch.where(
            masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
        )

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    # Keep the CLS token and mask the rest
    x_masked, ids_restore = random_masking(x[:, 1:, :], self.mask_ratio)
    x = torch.cat((x[:, :1, :], x_masked), dim=1)

    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )

    return x
