
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np
import transformers
from peft import get_peft_model, LoraConfig, AdaLoraConfig
from peft.peft_model import PeftModel
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, PreTrainedTokenizerFast
from model.embed import DataEmbedding,PatchEmbedding,PatchEmbedding_hf
from math import sqrt
from model.norm import Normalize
from tool.model_component import *

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dropout(x)
        return x
    
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Prompt_CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., K=10):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.K = K

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_norm = qk_norm
        # self.value_projection = nn.Linear(dim, d_keys * num_heads)

    def forward(self, prompt, enc_emd, word_emd):
        B, LP, D = prompt.shape
        B, LS, D = enc_emd.shape
        N, D = word_emd.shape

        # Compute query, key, and value
        qkv = self.qkv(torch.cat([prompt, enc_emd], dim=1))
        q, k, v = qkv.reshape(B, LP + LS, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.qk_norm:
            attn = attn / torch.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attended values
        x = (attn @ v).transpose(1, 2).reshape(B, LP + LS, D)

        # Project attended values
        x = self.proj(x)
        x = self.proj_drop(x)
        # x_norm = F.normalize(x, dim=-1)
        # word_emd_norm = F.normalize(word_emd, dim=-1)
        # Find top-k word em
        sim = torch.einsum('bld,vd->bv', x, word_emd)

        # Find top-k values and indices
        topk_values, topk_indices = torch.topk(sim, k=self.K, dim=-1)

        # Gather top-k word embeddings
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        # 从 word_emd 中收集 top-k 词嵌入向量
        output = torch.gather(word_emd.expand(B, -1, -1), 1, topk_indices)
        return output

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

def soft_hard_prompt_topk(prompt_embeddings,prompt_word_embeddings,K):
    if K==0:
        return prompt_embeddings
    
    B, LP, D = prompt_embeddings.shape

    sim = torch.einsum('bld,vd->bv', prompt_embeddings, prompt_word_embeddings)

    # Find top-k values and indices
    topk_values, topk_indices = torch.topk(sim, k=K, dim=-1)

    # Gather top-k word embeddings
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    # 从 word_emd 中收集 top-k 词嵌入向量
    prompt_embeddings = torch.gather(prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded)
    return prompt_embeddings, topk_indices

def soft_hard_prompt(prompt_embeddings,prompt_word_embeddings,K):
    if K==0:
        return prompt_embeddings
    
    B, LP, D = prompt_embeddings.shape

    sim = torch.einsum('bld,vd->bv', prompt_embeddings, prompt_word_embeddings)

    # Find top-k values and indices
    topk_values, topk_indices = torch.topk(sim, k=K, dim=-1)

    # Gather top-k word embeddings
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
    # 从 word_emd 中收集 top-k 词嵌入向量
    prompt_embeddings = torch.gather(prompt_word_embeddings.expand(B, -1, -1), 1, topk_indices_expanded)
    return prompt_embeddings

class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.in_channels=in_channels
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

    def forward(self, enc_out, enc_out_sgn):
        query = self.query(enc_out)
        key = self.key(enc_out_sgn)
        value = self.value(enc_out_sgn)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (self.in_channels ** 0.5)
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, value)
        fused_features = enc_out + attended_features
        return fused_features

class ReprogrammingLayer_single(nn.Module):
    def __init__(self, d_in, n_heads, d_keys=None, d_out=None, attention_dropout=0.1):
        super(ReprogrammingLayer_single, self).__init__()
        d_keys = d_keys or (d_in // n_heads)
        d_out = d_out or d_in
        
        self.query_projection = nn.Linear(d_in, d_keys * n_heads)
        self.key_projection = nn.Linear(d_out, d_keys * n_heads)
        self.value_projection = nn.Linear(d_out, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_out)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
    
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        
        return reprogramming_embedding
    
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        
        # 投影查询向量
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        
        # 投影键和值向量
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        
        # 注意力机制
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        
        # 输出投影
        return self.out_projection(out)

class ReprogrammingLayer_multi_layer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer_multi_layer, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.d_llm = d_llm
        
        # 第一层：输入d_model -> 输出d_llm
        self.layers.append(
            ReprogrammingLayer_single(
                d_model, n_heads, d_keys, d_llm, attention_dropout
            )
        )
        self.norm_layers.append(nn.LayerNorm(d_llm))
        
        # 后续层：输入d_llm -> 输出d_llm
        for _ in range(1, n_layers):
            self.layers.append(
                ReprogrammingLayer_single(
                    d_llm, n_heads, d_keys, d_llm, attention_dropout
                )
            )
            self.norm_layers.append(nn.LayerNorm(d_llm))
    
    def forward(self, target_embedding, source_embedding, value_embedding):
        x = target_embedding
        for i, (layer, norm_layer) in enumerate(zip(self.layers, self.norm_layers)):
            # 残差连接（第一层需要维度变换）
            residual = x
            if i == 0:  # 第一层维度变化
                residual = self.layers[0].out_projection(
                    self.layers[0].query_projection(residual))
            
            # 注意力变换
            x = layer(x, source_embedding, value_embedding)
            
            # 层归一化 + 残差连接
            x = norm_layer(x + residual)
        return x


         
    
class RadioLLM(nn.Module):

    def __init__(self, configs, yaml_config):
        super(RadioLLM, self).__init__()
        self.task_name = configs.task_name

        self.d_ff = configs.d_ff
        self.decoder_is=configs.decoder_is
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.is_LORA = configs.is_LORA

        if self.is_LORA:
            if configs.llm_model == 'GPT2':
                peft_config = LoraConfig(
                    r=8,  # LoRA矩阵的秩
                    lora_alpha=8,  # LoRA的缩放参数
                    lora_dropout=0.1,  # LoRA层的dropout率
                    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],  # 要应用LoRA的模块名称
                )
            elif configs.llm_model == 'LLAMA3.2':
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    target_modules=[
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.out_proj",
                        "mlp.fc1",
                        "mlp.fc2"
                    ],
                    lora_dropout=0.1,
                    bias="none",
                )
            elif configs.llm_model == 'BERT':
                peft_config = LoraConfig(
                    r=8,  # LoRA矩阵的秩
                    lora_alpha=8,  # LoRA的缩放参数
                    lora_dropout=0.1,  # LoRA层的dropout率
                    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
                )

        self.cls_tokens = nn.Parameter(torch.zeros(
            1, 1, self.d_llm))
        torch.nn.init.normal_(self.cls_tokens, std=.02)
        self.mask_tokens = nn.ParameterDict({})
        self.right_prob = configs.right_prob
        self.mask_ratio = {}
        self.pred_len = {}
        self.seq_len = {}
        self.head_nf = {}
        self.description = {}
        # self.output_projection = nn.Sequential(
        #     nn.Linear(self.d_llm, self.d_llm // 2, bias=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.d_llm // 2, configs.c_out, bias=True))
        self.output_projection = nn.ParameterDict({})
        self.predictor_ins = nn.ParameterDict({})
        self.predictor_cls = nn.ParameterDict({})
        self.high_freq_extract=nn.ParameterDict({})
        self.patch_embedding = nn.ParameterDict({}) 
        self.nmb_prototypes = {}
        self.patch_len = {}
        self.stride={}
        for i in range(len(yaml_config)):
            dataset_name = list(yaml_config.items())[i][1]['dataset_name']
            self.description[dataset_name] = list(yaml_config.items())[i][1]['content']
            self.min_mask_ratio = list(yaml_config.items())[i][1]['min_mask_ratio']
            self.max_mask_ratio = list(yaml_config.items())[i][1]['max_mask_ratio']
            self.mask_ratio[dataset_name] = [self.min_mask_ratio, self.max_mask_ratio]
            self.seq_len[dataset_name] = list(yaml_config.items())[i][1]['seq_len']
            self.pred_len[dataset_name] = self.seq_len[dataset_name]
            self.mask_tokens[dataset_name] = torch.zeros(1, 1, self.seq_len[dataset_name], 1)
            nn.init.normal_(self.mask_tokens[dataset_name], std=.02)
            if self.task_name in ['long_term_forecast', 'tsne', 'without_prompt', 'soft_hard_prompt',
                                  'soft_hard_prompt2', 'soft_hard_prompt3', 'classification']:
                self.patch_len[dataset_name] = list(yaml_config.items())[i][1]['patch_len']
                self.stride[dataset_name] = list(yaml_config.items())[i][1]['stride']
                self.patch_nums = int((self.seq_len[dataset_name] - self.patch_len[dataset_name]) / self.stride[dataset_name] + 2)
                self.head_nf[dataset_name] = self.d_ff * self.patch_nums
                self.output_projection[dataset_name] = FlattenHead(configs.enc_in, self.head_nf[dataset_name],
                                                                   self.pred_len[dataset_name],
                                                                   head_dropout=configs.dropout)
                self.nmb_prototypes[dataset_name] = list(yaml_config.items())[i][1]['numclass']
                self.predictor_ins[dataset_name] = nn.Sequential(
                    nn.Linear((self.d_llm - self.d_ff) * 2, (self.d_llm - self.d_ff) * 2),
                    nn.ReLU(),
                    nn.Linear((self.d_llm - self.d_ff) * 2, 256)
                )
                self.high_freq_extract[dataset_name] = High_freq_conv_3layer(d_model=configs.d_model,
                                        out_channel=self.d_llm, patch_len=self.patch_len[dataset_name]//4,
                                        stride=1, dropout=0)
                self.patch_embedding[dataset_name] = PatchEmbedding(configs.d_model, self.patch_len[dataset_name], self.stride[dataset_name], configs.dropout)

        # self.category_tokens = nn.Parameter(torch.zeros(
        #     1, 1, self.d_llm))
        # torch.nn.init.normal_(self.category_tokens, std=.02)
        self.llm_model_name = configs.llm_model
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(configs.llm_path)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LLAMA3.2':
            self.llama_config = LlamaConfig.from_pretrained(configs.llm_path)
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )


            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_path)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(configs.llm_path)

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2_w/o_pretrain':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_path)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model(config=self.gpt2_config)

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    configs.llm_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token


        self.attn_fusion = AttentionFusion(self.d_llm)
        self.dropout = nn.Dropout(configs.dropout)
        Attn = ProbAttention
        if self.task_name not in ['classificationwithout_prompt']:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(configs.decode_mask, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                                       self.d_llm, configs.n_heads, mix=configs.mix, attn=configs.attn),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            self.d_llm, configs.n_heads, mix=False),
                        self.d_llm,
                        configs.d_ff2,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(self.d_llm)
            )


        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer =ReprogrammingLayer_multi_layer(configs.d_model, configs.n_heads,configs.ReprogrammingLayer, self.d_ff, self.d_llm)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.K = configs.K
        if self.task_name in ['classification', 'classificationwithout_prompt']:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj_in = nn.Linear(self.d_llm, self.d_llm // 2)
            self.cross_att = CrossAttention(self.d_llm // 2)

            self.mlp_head = nn.Sequential(nn.Linear(self.d_llm*2, 128),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Linear(128, configs.numclass))
        elif self.task_name in ['soft_hard_prompt']:
            self.mapping_layer2 = nn.Linear(self.vocab_size, self.num_tokens)
            self.prompt_attn = Prompt_CrossAttention(self.d_llm, num_heads=configs.n_heads, qkv_bias=False,
                                                     qk_norm=False, attn_drop=configs.dropout,
                                                     proj_drop=configs.dropout, K=self.K)
        elif self.task_name in ['soft_hard_prompt2']:
            self.mapping_layer2 = nn.Linear(self.vocab_size, self.num_tokens)

        if self.is_LORA == False:
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            for name, param in self.llm_model.named_parameters():
                if 'lora' not in name:  # Assuming LoRA parameters have 'lora' in their names
                    param.requires_grad = False

    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample random masking.
        """
        N, V, L, D = x.shape  # batch, var, length, dim

        # Calculate mask ratios and lengths to keep for each sample in the batch
        mask_ratios = torch.rand(N, device=x.device) * \
                      (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)

        # Create a range tensor and compare with len_keeps for mask generation
        range_tensor = torch.arange(L, device=x.device).expand(N, L)
        mask = (range_tensor >= len_keeps.unsqueeze(1))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.float()

        return mask

    def continuous_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample continuous masking.
        """
        min_mask_ratio, max_mask_ratio = min_mask_ratio / 2, max_mask_ratio / 2
        N, V, L, D = x.shape  # batch, var, length, dim

        # Initialize mask
        mask = torch.zeros([N, L], device=x.device)

        for i in range(N):
            # Determine the length of the masked region for this sample
            mask_length = int(torch.randint(int(min_mask_ratio * L), int(max_mask_ratio * L) + 1, (1,)).item())

            # Randomly choose the start point of the masked region
            start = torch.randint(0, L - mask_length + 1, (1,)).item()

            # Apply mask
            mask[i, start:start + mask_length] = 1

        return mask.float()

    def right_masking(self, x, min_mask_ratio, max_mask_ratio):
        N, V, L, D = x.shape  # batch, var, length, dim

        # Randomly choose a mask ratio for each sample within the specified range
        mask_ratios = torch.rand(N, device=x.device) * \
                      (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        # Binary mask creation without a for loop
        len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
        indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
        mask = indices >= len_keeps_matrix
        mask = mask.float()

        return mask

    def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        if torch.rand(1).item() > right_prob:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        else:
            return self.continuous_masking(x, min_mask_ratio, max_mask_ratio)

    def forward(self, x_enc, enable_mask=False, dataset_name=None):
        if self.task_name in ['long_term_forecast', 'soft_hard_prompt', 'soft_hard_prompt2', 'soft_hard_prompt3']:
            if enable_mask:
                dec_out, mask_seq = self.forecast(x_enc, enable_mask, dataset_name)

                return dec_out[:, -self.pred_len[dataset_name]:, :], mask_seq
            else:
                dec_out = self.forecast(x_enc, enable_mask, dataset_name)

                return dec_out[:, -self.pred_len[dataset_name]:, :]
        if self.task_name == 'without_prompt':
            dec_out = self.forecast_without_prompt(x_enc)
            return dec_out[:, -self.pred_len[dataset_name]:, :]
        if self.task_name == 'classification':
            out = self.classific(x_enc, enable_mask, dataset_name)
            return out
        if self.task_name == 'classificationwithout_prompt':
            out = self.classific_without_prompt(x_enc)
            return out
        if self.task_name == 'classificationwithenc':
            out = self.classific_withenc(x_enc, enable_mask, dataset_name)
            return out
        if self.task_name == 'tsne':
            dec_out_q = self.tsne(x_enc, enable_mask, dataset_name)
            # dec_out_q = self.pool(dec_out_q.transpose(1, 2))
            # dec_out_q = dec_out_q.view(dec_out_q.size(0), -1)

            # dec_out_q= nn.functional.normalize(dec_out_q, dim=1)
            return dec_out_q
        return None

    def forecast(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)
        enc_out_sgn_1 = self.high_freq_extract[dataset_name](x_enc)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d',n=n_vars)

        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # enc_out = enc_out_sgn_1
        if self.task_name in ['soft_hard_prompt']:
            prompt_word_embeddings = self.mapping_layer2(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = self.prompt_attn(prompt_embeddings, enc_out, prompt_word_embeddings)
        elif self.task_name in ['soft_hard_prompt2']:
            prompt_word_embeddings = self.mapping_layer2(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        elif self.task_name in ['soft_hard_prompt3']:
            prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = dec_out[:, :, :, self.prompt_length:-1]  # B,var,dim,len
        if self.decoder_is:
            dec_out_1 = einops.rearrange(dec_out, 'b n d l -> b (n l) d')
            dec_out = self.decoder(dec_out_1, dec_out_1, x_mask=None, cross_mask=None)
            dec_out = einops.rearrange(dec_out, 'b (n l) d -> b n d l', n=2)
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = self.output_projection[dataset_name](dec_out)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out.permute(0, 2, 1)
        if enable_mask:
            return dec_out, mask_seq
        else:
            return dec_out

    def forecast_without_prompt(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding[dataset_name](x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)

        llama_enc_out = torch.cat([enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, :-1])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def tsne(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)
        enc_out_sgn_1 = self.high_freq_extract[dataset_name](x_enc)

        dec_out_con = self.pool(enc_out_sgn_1)
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        # enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)
        # enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        # enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        # cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        # self.prompt_length = prompt_embeddings.shape[1]
        # llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = torch.reshape(
        #     dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # dec_out = dec_out[:, :, :, self.prompt_length:-1]  # B,var,dim,len
        # dec_out_con = einops.rearrange(dec_out[:, :, self.d_ff:], 'b n d l -> b (n d) l')
        # dec_out_con = self.pool(dec_out_con)
        # dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        # dec_out_con = self.predictor_ins[dataset_name](dec_out_con)

        return dec_out_con

    def classific(self, x_enc, enable_mask=False, dataset_name=None, encoder_k_is=False):
        x_enc_ori=x_enc
        x_enc = x_enc.permute(0, 2, 1)

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()

        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)

        enc_out_sgn, n_vars = self.patch_embedding[dataset_name](x_enc)
        enc_out_sgn_1 = self.high_freq_extract[dataset_name](x_enc_ori)
        enc_out_sgn_1 = einops.rearrange(enc_out_sgn_1, 'b (n d) l -> (b n) l d', n=n_vars)

        enc_out = self.reprogramming_layer(enc_out_sgn, source_embeddings, source_embeddings)
        enc_out = self.attn_fusion(enc_out, enc_out_sgn_1)
        # enc_out = enc_out_sgn_1
        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = einops.rearrange(dec_out, '(b n) l d -> b l (n d)', n=2)
        dec_out_con= dec_out[:, self.prompt_length:-1, :]
        dec_out_con=self.pool(dec_out_con.transpose(1, 2))
        dec_out_con = dec_out_con.view(dec_out_con.size(0), -1)
        dec_out_con = self.mlp_head(dec_out_con)


        return dec_out_con

    def classific_withenc(self, x_enc, enable_mask=False, dataset_name=None):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            if enable_mask == False:
                task_prompt = f"Task description: denoising a radio signal based on {str(self.pred_len)} samples with noise; "
            else:
                task_prompt = f"Task description: recovering a missing radio signal based on {str(self.pred_len)} samples; "
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description[dataset_name]}"
                f"{task_prompt}"
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        if enable_mask:
            x_enc = x_enc.unsqueeze(dim=-1)
            mask = self.choose_masking(x_enc, self.right_prob,
                                       self.mask_ratio[dataset_name][0], self.mask_ratio[dataset_name][1])
            mask_repeat = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
            mask_repeat = mask_repeat.repeat(1, x_enc.shape[1], 1, x_enc.shape[-1])
            x_enc = x_enc * (1 - mask_repeat) + self.mask_tokens[dataset_name] * mask_repeat
            mask_seq = mask
            x_enc = x_enc.squeeze(dim=-1)
        enc_out, n_vars = self.patch_embedding[dataset_name](x_enc)

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        prompt_word_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        prompt_embeddings = soft_hard_prompt(prompt_embeddings, prompt_word_embeddings, self.K)
        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)
        self.prompt_length = prompt_embeddings.shape[1]
        enc_out = torch.cat([prompt_embeddings, enc_out, cls_tokens], dim=1)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars * enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.transpose(1, 2)
        if enable_mask:
            return enc_out, cls_tokens
        else:
            return enc_out

    def classific_without_prompt(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding[dataset_name](x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        cls_tokens = self.cls_tokens.repeat(enc_out.shape[0], 1, 1)

        llama_enc_out = torch.cat([enc_out, cls_tokens], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = einops.rearrange(dec_out, 'b n l c -> b (n l) c')
        # dec_out= dec_out.permute(0, 2, 1)
        # dec_out =self.pool(dec_out)
        # latent = torch.flatten(dec_out, 1)
        # out = self.mlp_head(latent)

        x = self.proj_in(dec_out)
        B, V, L, C = x.shape
        x = x.view(-1, L, C)
        cls_token = x[:, -1:]
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)
        cls_token = einops.rearrange(cls_token, 'b n l c -> b (n l) c')
        cls_token = cls_token.permute(0, 2, 1)
        cls_token = self.pool(cls_token)
        cls_token = torch.flatten(cls_token, 1)
        out = self.mlp_head(cls_token)
        return out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


if __name__ == '__main__':
    import argparse
    import yaml
    from thop import profile
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v == 'True':
            return True
        if v == 'False':
            return False


    parser = argparse.ArgumentParser(description='Time-LLM')
    # basic config
    parser.add_argument('--task_name', type=str, required=False,
                        default='soft_hard_prompt3',
                        help='task name, options:[soft_hard_prompt3, tsne, classification, without_prompt]')
    parser.add_argument("--decoder_is", type=str2bool, default=False)
    # forecasting task
    parser.add_argument('--prompt_domain', type=str2bool, default=True)
    parser.add_argument('--is_LORA', type=str2bool, default=True)
    parser.add_argument('--numclass', type=int, default=11)
    # model define
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--nmb_prototypes', type=int, default=300, help='nmb_prototypes')
    parser.add_argument('--T', type=float, default=0.07, help='tempeture')
    parser.add_argument('--epsilon', type=float, default=0.05, help=' ')
    parser.add_argument('--sinkhorn_iterations', type=int, default=3, help=' ')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--patch_len', type=int, default=4, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--llm_path', type=str, default= r'D:\planedemo\SignalGPT\pretrain_model\gpt2',
                        help='LLM_model_path')

    parser.add_argument('--llm_dim', type=int, default='768',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768; LLama3.2 1B:2048; Lama3.2 3B:3072;
    parser.add_argument('--ReprogrammingLayer', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--right_prob', type=float, default=0.5)

    parser.add_argument('--K', type=int, default=7, help='top K prompt')

    parser.add_argument("--decode_mask", type=str2bool, default=True)
    parser.add_argument("--mix", type=str2bool, default=True)
    parser.add_argument("--attn", type=str, default='prob')
    parser.add_argument('--d_ff2', type=int, default=1024, help='dimension of fcn')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取YAML文件
    with open(r'D:\planedemo\毕设论文代码\第三章\script\multi_task_pretrain.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)
    model = RadioLLM(args, yaml_config=yaml_config)
    sequence_length = 128
    x_enc = torch.randn(args.batch_size, args.enc_in, sequence_length)


    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
        model = model.to(device)
        
    else:
        model = model.cuda()

    out=model(x_enc.cuda(), enable_mask=False, dataset_name='RML2016a+b_total_snr')
    for o in out:
        print(o.shape)
    # macs, params, layer_info = profile(model, inputs=(x_enc.cuda(),), verbose=False, ret_layer_info=True)
    # print(f"Model Parameters: {params / 1e6:.2f} M")

    def count_parameters_in_MB(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_MB = total_params * 4 / (1024 ** 2)
        trainable_params_MB = trainable_params * 4 / (1024 ** 2)
        ratio = trainable_params / total_params if total_params > 0 else 0
        return total_params_MB, trainable_params_MB, ratio


    # Example usage:

    total_params_MB, trainable_params_MB, ratio = count_parameters_in_MB(model)

    print(f"Total parameters: {total_params_MB:.2f} MB")
    print(f"Trainable parameters: {trainable_params_MB:.2f} MB")
    print(f"Ratio of trainable to total parameters: {ratio:.4f}")


