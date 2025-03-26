import os
import types
from typing import Callable
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from transformers import AutoTokenizer, logging, AutoModelForCausalLM

from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)

from dataset.constants_val import HF_CKPT_CACHE_DIR
from backbones.utils import (
    get_tokenizer,
    masked_only_prepare_tokens_with_masks,
    _parse_dinov2_model_name,
    _make_dinov2_model,
)

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_API_TOKEN = "<replace-with-your-hf-api-token>"


class GlobalEmbedding(nn.Module):
    def __init__(
        self, input_dim: int = 768, hidden_dim: int = 2048, output_dim: int = 512
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False),  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class DinoEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14_reg_lc",
        img_size: int = 224,
        text_feat_dim: int = 768,
        output_dim: int = 512,
        hidden_dim: int = 2048,
        freeze_vit: bool = False,
        pretrained: bool = True,
        linear_proj: bool = True,
        num_freeze_blocks: int = 0,
        vit_grad_ckpt: bool = False,
        is_det: bool = False,
        **kwargs,
    ):
        super(DinoEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim
        self.is_det = is_det

        if "dinov2" in model_name:
            arch_name, pretrained, num_register_tokens, patch_size = (
                _parse_dinov2_model_name(model_name)
            )
            self.model = _make_dinov2_model(
                arch_name=arch_name,
                patch_size=patch_size,
                pretrained=pretrained,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True,
                interpolate_offset=0.0,
                grad_ckpt=vit_grad_ckpt,
            )
        else:
            print(self.model_name)
            raise NotImplementedError

        self.model.mask_token.requires_grad = False  # never train the mask token

        self.feature_dim = self.model.embed_dim

        if linear_proj:
            self.global_embed = nn.Linear(self.feature_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.feature_dim, hidden_dim, output_dim
            )

        # Unused
        self.local_embed = LocalEmbedding(self.feature_dim, hidden_dim, output_dim)

        if freeze_vit:
            print("Freezing vit model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False
            for param in self.local_embed.parameters():
                param.requires_grad = False

        if num_freeze_blocks > 0:
            pass  # TODO

        if self.is_det:
            self.fpn_down_conv = nn.Conv2d(
                self.feature_dim,
                2 * self.feature_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            )  # bsz x 2C x H/2 x W/2
            self.fpn_conv = nn.Conv2d(
                self.feature_dim, self.feature_dim, kernel_size=1, stride=1, padding=0
            )  # bsz x C x H x W
            self.fpn_up_conv = nn.ConvTranspose2d(
                self.feature_dim,
                self.feature_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )  # bsz x C/2 x 2H x 2W
            self.filters = [
                self.feature_dim // 2,
                self.feature_dim,
                2 * self.feature_dim,
            ]

    def det_forward(self, x):
        ret = self.model(x, is_training=True)
        x = ret["x_norm_patchtokens"].contiguous()  # B, S, C
        x = x.permute(0, 2, 1)  # B, C, S
        H = int(x.shape[2] ** 0.5)
        B, C = x.shape[:2]
        x = x.view(B, C, H, H)  # B, C, H, W
        x0 = self.fpn_down_conv(x)
        x1 = self.fpn_conv(x)
        x2 = self.fpn_up_conv(x1)
        return x2, x1, x0

    def vit_forward(self, x):
        return self.model(x, is_training=True)

    def forward(self, x, get_local=False):
        if self.is_det:
            return self.det_forward(x)
        ret = self.vit_forward(x)
        return (
            ret["x_norm_clstoken"].contiguous(),
            ret["x_norm_patchtokens"].contiguous(),
            ret["x_prenorm"].contiguous(),
        )


class CausalLMEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        emb_dim: int = 2560,
        output_dim: int = 512,
        hidden_dim: int = 2048,
        freeze_llm: bool = True,
        agg_tokens: bool = False,
        peft: str = None,
        grad_ckpt: bool = False,
        llm_type: str = "gpt",
        ctx_prompt: bool = False,
        img_ctx_prompt: bool = False,
        linear_proj: bool = True,
        unlock_ln: bool = False,
        num_freeze_blocks: int = 0,
        total_steps: int = 40000,
        avg_sent_feat: bool = False,
    ):
        super(CausalLMEncoder, self).__init__()

        self.llm_type = llm_type
        if self.llm_type == "gpt":
            self.llm_name = "stanford-crfm/BioMedLM"
            model_param = {
                # "torch_dtype": torch.bfloat16,
                # "low_cpu_mem_usage": True,
            }
            emb_dim = 2560
        elif self.llm_type == "llama":
            self.llm_name = "epfl-llm/meditron-7b"
            model_param = {
                "torch_dtype": torch.bfloat16,
                # "load_in_4bit": True,
                # "bnb_4bit_compute_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                # "device_map": "auto"
            }
            emb_dim = 4096
        elif self.llm_type == "llama2":
            self.llm_name = "meta-llama/Llama-2-7b-hf"
            model_param = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            emb_dim = 4096
        elif self.llm_type == "llama3":
            self.llm_name = "meta-llama/Meta-Llama-3-8B"
            model_param = {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "cache_dir": HF_CKPT_CACHE_DIR,
            }
            emb_dim = 4096
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_llm = freeze_llm
        self.agg_tokens = agg_tokens
        self.ctx_prompt = ctx_prompt
        self.img_ctx_prompt = img_ctx_prompt
        self.avg_sent_feat = avg_sent_feat
        # self.max_sent_num = 10

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name, token=MY_API_TOKEN, **model_param
        )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(self.llm_type)

        # Update vocab embedding with new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        # Remove the LM head
        self.model.lm_head = nn.Identity()

        if peft:
            self.model = self.get_peft_model(peft, total_steps)

        # Default CKPT failed in forward pass
        if grad_ckpt:
            if self.llm_type == "gpt":
                self.model.transformer.gradient_checkpointing_enable()
            elif self.llm_type in ["llama", "llama2", "llama3"]:
                self.model.model.gradient_checkpointing_enable()

        if self.freeze_llm is True:
            print("Freezing llm model")
            for param in self.model.parameters():
                param.requires_grad = False

        if linear_proj:
            self.global_embed = nn.Linear(self.embedding_dim, output_dim)
        else:
            self.global_embed = GlobalEmbedding(
                self.embedding_dim, hidden_dim, self.output_dim
            )
        # Unused
        self.local_embed = LocalEmbedding(
            self.embedding_dim, hidden_dim, self.output_dim
        )
        self.global_embed = self.global_embed.to(self.model.dtype)
        self.local_embed = self.local_embed.to(self.model.dtype)

        if self.ctx_prompt or self.img_ctx_prompt:
            print("Freezing full llm model")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.global_embed.parameters():
                param.requires_grad = False
            for param in self.local_embed.parameters():
                param.requires_grad = False

        if unlock_ln:
            print("Unlocking LayerNorm within pre-trained LLM")
            for name, param in self.model.named_parameters():
                if "ln" in name:
                    param.requires_grad = True

        if num_freeze_blocks > 0:
            if self.llm_type == "gpt":
                print("Freeze first {} blocks in GPT model".format(num_freeze_blocks))
                for name, param in self.model.named_parameters():
                    for i in range(num_freeze_blocks):
                        if f"h.{i}." in name:
                            param.requires_grad = False
            elif self.llm_type in ["llama", "llama2", "llama3"]:
                # TODO
                pass

    def get_peft_model(self, peft, total_steps=40000):
        print(f"Using PEFT: {peft}")
        if self.llm_type == "gpt":
            target_modules = ["c_attn", "mlp.c_proj"]
            feedforward_modules = ["mlp.c_proj"]
        elif self.llm_type in ["llama", "llama2", "llama3"]:
            target_modules = ["q_proj", "v_proj"]
            feedforward_modules = ["down_proj"]

        inference_mode = self.ctx_prompt or self.img_ctx_prompt
        if peft == "ia3":
            config = IA3Config(
                peft_type="ia3",
                task_type=TaskType.CAUSAL_LM,
                inference_mode=inference_mode,
                target_modules=target_modules,
                feedforward_modules=feedforward_modules,
            )
        elif peft == "lora":
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=inference_mode,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        elif peft == "prefix":
            config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=inference_mode,
                num_virtual_tokens=20,
            )
        elif peft == "adalora":
            config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=inference_mode,
                target_r=8,
                init_r=12,
                tinit=500,
                tfinal=1500,
                beta1=0.85,
                beta2=0.85,
                total_step=total_steps,
            )
        peft_model = get_peft_model(self.model, config)
        print(peft_model.print_trainable_parameters())
        return peft_model

    def find_last_word_token(self, embeddings, caption_ids):
        """
        :param embeddings: bz, 1, S, C
        :param caption_ids: bz, S
        """

        bz, _, _, _ = embeddings.shape
        last_word_tokens = []
        pad_token = self.tokenizer.pad_token_id
        # print(caption_ids.shape)
        # print(embeddings.shape)
        for i in range(bz):
            # print(caption_ids[i, :])
            # Need to consider the prepending Tokens
            last_word_idx = 0
            for j in range(1, len(caption_ids[i, :]) + 1):
                # First padding token
                if caption_ids[i, -j] == pad_token:
                    last_word_idx -= 1
                    continue
            # last_word_idx = torch.argwhere(caption_ids[i, :] == eos_token)[0][0].item()
            # print(caption_ids[i, last_word_idx - 10:])
            # print(last_word_idx, caption_ids[i, last_word_idx])
            last_word_tokens.append(embeddings[i, 0, last_word_idx, :].unsqueeze(0))
        return torch.stack(last_word_tokens, dim=0)

    def find_all_sep_tokens(self, embeddings, caption_ids):
        """
        :param embeddings: bz, 1, S, C
        :param caption_ids: bz, S
        """
        bz, _, _, _ = embeddings.shape
        sep_tokens = []
        sep_token = self.tokenizer.sep_token_id
        for i in range(bz):
            sep_token_idx = torch.argwhere(caption_ids[i, :] == sep_token).squeeze()
            if self.avg_sent_feat:
                prev_idx = 0
                sent_feat = []
                for sep_idx in sep_token_idx:
                    # use average of sentence feature as final feature
                    cur_feats = (
                        embeddings[i, 0, prev_idx:sep_idx, :].mean(dim=0).contiguous()
                    )  # C
                    sent_feat.append(cur_feats)
                    prev_idx = sep_idx
                sent_feat = torch.stack(sent_feat, dim=0).unsqueeze(0)
            else:
                sent_feat = embeddings[i, 0, sep_token_idx, :].unsqueeze(0).contiguous()
            sep_tokens.append(sent_feat)  # 1, S, C
        return sep_tokens

    def forward(self, ids, attn_mask, inputs_embeds=None, get_local=False, **kwargs):
        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)
        if self.ctx_prompt or self.img_ctx_prompt:
            # Use input embeddings instead of ids
            assert inputs_embeds != None
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                output_attentions=True,
                return_dict=True,
            )
        else:
            outputs = self.model(
                input_ids=ids,
                attention_mask=attn_mask,
                output_attentions=True,
                return_dict=True,
            )
        target_dtype = self.model.dtype

        last_layer_attn = (
            outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).to(target_dtype)
        )
        all_feat = outputs.logits.unsqueeze(1).to(target_dtype)

        sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]
        last_atten_pt = last_layer_attn.contiguous()

        # Causal LM: only the last word token is used as the report feature
        report_feat = self.find_last_word_token(all_feat, ids).contiguous()
        word_feat = all_feat[:, :, :].contiguous()
        sents_feat = self.find_all_sep_tokens(all_feat, ids)

        if self.last_n_layers == 1:
            report_feat = report_feat[:, 0]
            word_feat = word_feat[:, 0]

        return report_feat, word_feat, last_atten_pt, sents_feat
