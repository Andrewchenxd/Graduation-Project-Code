import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import math
from model.CVCNN import ComplexConv
import einops
import copy
from tool.StandardNorm import Normalize
import numpy as np

from math import sqrt
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

class high_freq_extract(nn.Module):
    def __init__(self, in_channels=128, out_channels=768, dropout=0.1):
        super(high_freq_extract, self).__init__()
        self.dropout_rate = dropout
        self.conv1 = ComplexConv(in_channels=in_channels, out_channels=out_channels//8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_channels//4)

        self.conv2 = ComplexConv(in_channels=out_channels//8, out_channels=out_channels//4, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_channels//2)

        self.conv3 = ComplexConv(in_channels=out_channels//4, out_channels=out_channels//2, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=out_channels)

        self.convres = ComplexConv(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, padding=1)
        self.batchnormres = nn.BatchNorm1d(num_features=out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sgn):
        x = self.conv1(sgn)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x) 

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x) 

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout(x) 

        res = self.convres(sgn)
        res = F.relu(res)
        res = self.batchnormres(res)
        res = self.dropout(res) 

        x = x + res
        return x

class High_freq_conv(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        return x


class High_freq_conv_3layer(nn.Module):
    def __init__(self, d_model,out_channel, patch_len, stride, dropout):
        super(High_freq_conv_3layer, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.projhead_dim=out_channel*2
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding =nn.Conv2d(1, d_model, kernel_size=(1, patch_len), stride=(1,stride), padding=(0, stride), bias=False)

        self.high_freq_exter=high_freq_extract(in_channels=d_model,out_channels=d_model*2,dropout=dropout)
        self.high_freq_exter2 = high_freq_extract(in_channels=d_model, out_channels=out_channel * 2,dropout=dropout)
        self.high_freq_exter3 = high_freq_extract(in_channels=out_channel, out_channels=out_channel * 2,dropout=dropout)
        self.pool=nn.MaxPool1d(kernel_size=2)
        self.drop=nn.Dropout(dropout)

    def forward(self, x):
        x = self.padding_patch_layer(x)
        x=torch.unsqueeze(x,1)
        x = self.value_embedding(x)
        x = self.drop(x)
        b,n,d,l=x.shape
        x = einops.rearrange(x, 'b d n l -> b (n d) l')
        x=self.high_freq_exter(x)
        x=self.pool(x)
        x = self.high_freq_exter2(x)
        x = self.pool(x)
        x = self.high_freq_exter3(x)
        x = self.pool(x)
        return x
    
class SWT_RAG(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder,nmb_prototypes=128, K=65536, m=0.99, T=0.1,epsilon=0.05,sinkhorn_iterations=10,redution='resize'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SWT_RAG, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.epsilon=epsilon
        self.sinkhorn_iterations=sinkhorn_iterations
        self.redution=redution

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = copy.deepcopy(base_encoder)
        self.encoder_k = copy.deepcopy(base_encoder)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.prototypes = nn.Linear(base_encoder.projhead_dim, nmb_prototypes,bias=False)
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            # param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def distributed_sinkhorn(self,out,world_size=1):
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def contrastive_loss(self, scores_q, scores_k):
        temperature = self.T
        q_k = self.distributed_sinkhorn(scores_k)
        p_q=F.log_softmax(scores_q/temperature, dim=1)
        loss=-torch.mean(torch.sum(q_k*p_q, dim=1))
        # loss = -torch.mean(torch.sum(q_k * p_t, dim=1))
        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        '''
        Input:
            x1: a batch of query images  noise
            x2: a batch of key images    time warp/freq scale
        Output:
            q1, q2, k1, k2
        '''
        code_q = self.encoder(x2)  # keys: NxC
        code_q = self.pool(code_q).squeeze(-1)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            code_k = self.encoder_k(x1)
            code_k = self.pool(code_k).squeeze(-1)

        
        q1_copy=code_q.detach()
        q2_copy=code_k.detach()
        # compute key features
        B=code_q.shape[0]
        score=torch.cat((code_q,code_k),0)
        score=self.prototypes(score)
        score_q, score_k = torch.split(score, B, 0)
        loss=self.contrastive_loss(score_q, score_k),q1_copy,q2_copy


        return  loss

class MoCoV3_RAG(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoV3_RAG, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = copy.deepcopy(base_encoder)
        self.encoder_k = copy.deepcopy(base_encoder)
        self.normalize_layers = Normalize(2, affine=False)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.mlp = mlp

        if self.mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.projhead_dim
            self.encoder_q_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))
            self.encoder_k_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_mlp))
        else:
            self.encoder_q_fc = nn.Identity()
            self.encoder_k_fc = nn.Identity()


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=logits.device)  # 使用 logits 的设备
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2):
        '''
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1, q2, k1, k2
        '''
        x1 = self.normalize_layers(x1.permute(0, 2, 1), 'norm').permute(0, 2, 1).contiguous()
        x2 = self.normalize_layers(x2.permute(0, 2, 1), 'norm').permute(0, 2, 1).contiguous()
        q1 = self.encoder_q(x1)
        q2 = self.encoder_q(x2)
        q1_copy=q1.detach()
        q2_copy=q2.detach()


        q1 = self.pool(q1).squeeze(-1)
        q2 = self.pool(q2).squeeze(-1)
        
        q1=self.encoder_q_fc(q1)
        q2=self.encoder_q_fc(q2)

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        
        

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k1, k2 = self.encoder_k(x1), self.encoder_k(x2)  # keys: NxC
            k1 = self.pool(k1).squeeze(-1)
            k2 = self.pool(k2).squeeze(-1)
            
            k1=self.encoder_k_fc(k1)
            k2=self.encoder_k_fc(k2)
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1),q1_copy,q2_copy




class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class FreqAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FreqAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        queries_freq =torch.abs(torch.fft.fft(queries))
        keys_freq =torch.abs(torch.fft.fft(keys))
        values_freq = torch.abs(torch.fft.fft(values))

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scores_top_freq, index_freq = self._prob_QK(queries_freq, keys_freq, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        context_freq = self._get_initial_context(values_freq, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        context_freq, _ = self._update_context(context_freq, values_freq, scores_top_freq, index_freq, L_Q, attn_mask)
        context_freq=torch.abs(torch.fft.ifft(context_freq))
        return context.transpose(2, 1).contiguous(),context_freq.transpose(2, 1).contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False,attn='prob'):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
        self.attn=attn
        if self.attn=='freq':
            self.pro_linear=nn.Linear(d_values * n_heads,d_values * n_heads)
            self.pro_linear_freq = nn.Linear(d_values * n_heads, d_values * n_heads)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        if self.attn!='freq':
            out, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask
            )
            if self.mix:
                out = out.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)
        else:
            out,out_freq, attn = self.inner_attention(
                queries,
                keys,
                values,
                attn_mask
            )
            if self.mix:
                out = out.transpose(2, 1).contiguous()
                out_freq = out_freq.transpose(2, 1).contiguous()
            out = out.view(B, L, -1)
            out_freq=out_freq.view(B, L, -1)
            out=self.pro_linear(out)
            out_freq = self.pro_linear_freq(out_freq)
            out=out+out_freq+values.view(B, L, -1)


        return self.out_projection(out), attn


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)



class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

if __name__=="__main__":
    base_model=High_freq_conv_3layer(d_model=64,out_channel=128, patch_len=2, stride=1, dropout=0.1)
    input=torch.randn(32,2,128)
    # Model=SWT_RAG(base_encoder=base_model,nmb_prototypes=128)
    Model=MoCoV3_RAG(base_encoder=base_model,mlp=True)
    output=Model(input,input)
    print(output)
