import os
import torch
import torch.nn as nn
from einops import rearrange
from .med import BertModel
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging
from .utils import get_tokenizer

logging.set_verbosity_error()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


class BertEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: BertTokenizer = None,
        emb_dim: int = 768,
        output_dim: int = 128,
        hidden_dim: int = 2048,
        freeze_llm: bool = True,
        agg_tokens: bool = False,
        linear_proj: bool = True,
        num_freeze_blocks: int = 0,
        **kwargs
    ):
        super(BertEncoder, self).__init__()
        self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        self.llm_type = "bert"
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_llm = freeze_llm
        self.agg_tokens = agg_tokens
        # self.max_sent_num = 10

        self.config = BertConfig.from_json_file(
            os.path.join(BASE_DIR, "./bert_config.json")
        )
        self.model = BertModel.from_pretrained(
            self.bert_type,
            config=self.config,
            add_pooling_layer=False,
        )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_tokenizer(self.llm_type)

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        if self.freeze_llm is True:
            print("Freezing BERT model")
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
        self.global_embed.to(self.model.dtype)
        self.local_embed.to(self.model.dtype)

        if num_freeze_blocks > 0:
            # TODO
            pass

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        """
        :param embeddings: bz, 1, S, 768
        :param caption_ids: bz, S
        :param last_layer_attn: bz, 111
        """
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(
            embeddings, caption_ids, last_layer_attn
        ):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0)
            )
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt

    def find_all_sep_tokens(self, embeddings, caption_ids):
        """
        :param embeddings: bz, S, C
        :param caption_ids: bz, S
        """
        bz, _, _ = embeddings.shape
        sep_tokens = []
        sep_token = self.tokenizer.sep_token_id
        for i in range(bz):
            sep_token_idx = torch.argwhere(caption_ids[i, :] == sep_token).squeeze()
            sep_tokens.append(
                embeddings[i, sep_token_idx, :].unsqueeze(0).contiguous()
            )  # S, C
        return sep_tokens

    def forward(self, ids, attn_mask, token_type, get_local=False):
        if len(ids.shape) == 1:
            ids = ids.unsqueeze(0)
        outputs = self.model(
            ids,
            attn_mask,
            token_type,
            output_attentions=True,
            return_dict=True,
            mode="text",
        )
        target_dtype = self.model.dtype

        last_layer_attn = (
            outputs.attentions[-1][:, :, 0, 1:].mean(dim=1).to(target_dtype)
        )
        all_feat = outputs.last_hidden_state.unsqueeze(1).to(target_dtype)

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, ids, last_layer_attn
            )
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]
            last_atten_pt = last_layer_attn.contiguous()
        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0]

        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()
        sents_feat = self.find_all_sep_tokens(all_feat, ids)

        return report_feat, word_feat, last_atten_pt, sents_feat
