from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ..model.config import Config
from ..util.device_copy import to_device
from ..util.tensor import to2
from ..util.rope import RoPE, RopeSettings, RopeStyle
from ..constants import PAGE_SIZE
from . import Module, Linear, RMSNorm, GatedMLP, BlockSparseMLP
from .attn import get_for_device
from .ngram_embedding import NGramEmbedding
from .ple import PLELayer, PLELayerState

"""
Hy4 flash (Tencent 'hy_v4_flash') building blocks: independent hyper-connections (iHC), gated
MLA with sliding-window or grouped compressed-indexer sparse attention, LatentMoE and Engram.

Reference: transformers/models/hy_v4_flash + hy_v4 (Tencent's released model code) and the vLLM
hy_v4_flash PR. Tensor names are the checkpoint's: model.layers.N...
"""


class HyV4HyperConnection(Module):
    """iHC site: collapse hc_mult streams with a pre-gate, publish a post-gate for the update."""

    def __init__(self, config: Config, key: str, hc_mult: int, hidden_size: int,
                 magnitude: float, eps: float, rms_norm_eps: float):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.magnitude = magnitude
        self.hc_eps = eps
        self.rms_eps = rms_norm_eps
        self.fn = None
        self.base = None
        self.scale = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}.hc_fn", device, no_defer = True, arena = False).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}.hc_base", device, no_defer = True, arena = False).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}.hc_scale", device, no_defer = True, arena = False).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.base = self.scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.hc_fn": self.fn.contiguous(),
            f"{self.key}.hc_base": self.base.contiguous(),
            f"{self.key}.hc_scale": self.scale.contiguous(),
        }

    @override
    def optimizer_targets(self):
        return []

    @override
    def optimizer_targets(self):
        return []

    def forward(self, x: torch.Tensor, params: dict, out_dtype = None):
        return self.mix(x)[1]

    def mix(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hc_mult
        flat = streams.float().flatten(2)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim = True) + self.rms_eps)
        mixes = F.linear(flat, self.fn) * rms
        pre_b, post_b = self.base.split(h, dim = -1)
        pre_scale, post_scale = self.scale.unbind(0)
        pre_logits, post_logits = mixes.split(h, dim = -1)
        pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps
        post = self.magnitude * torch.sigmoid(post_logits * post_scale + post_b) + self.hc_eps
        out = torch.sum(pre.unsqueeze(-1) * streams, dim = 2)
        return post, out

    def mix_norm(self, streams: torch.Tensor, params: dict, norm: RMSNorm):
        post, y = self.mix(streams)
        return post, None, y, False

    def apply_(self, streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor,
               comb, params: dict) -> torch.Tensor:
        return post.to(y.dtype).unsqueeze(-1) * y.unsqueeze(-2) + streams


class HyV4HyperHead(Module):
    def __init__(self, config: Config, key: str, hc_mult: int, hidden_size: int,
                 eps: float, rms_norm_eps: float):
        super().__init__(config = config, key = key, qmap = None)
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        self.hc_eps = eps
        self.rms_eps = rms_norm_eps
        self.fn = None
        self.base = None
        self.scale = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.fn = stc.get_tensor(f"{self.key}.hc_head_fn", device, no_defer = True, arena = False).float().contiguous()
        self.base = stc.get_tensor(f"{self.key}.hc_head_base", device, no_defer = True, arena = False).float().contiguous()
        self.scale = stc.get_tensor(f"{self.key}.hc_head_scale", device, no_defer = True, arena = False).float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.fn = self.base = self.scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.hc_head_fn": self.fn.contiguous(),
            f"{self.key}.hc_head_base": self.base.contiguous(),
            f"{self.key}.hc_head_scale": self.scale.contiguous(),
        }

    @override
    def optimizer_targets(self):
        return []

    @override
    def optimizer_targets(self):
        return []

    def forward(self, x: torch.Tensor, params: dict | None = None, out_dtype = None) -> torch.Tensor:
        flat = x.float().flatten(2)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim = True) + self.rms_eps)
        mixes = F.linear(flat, self.fn) * rms
        pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim = 2)


def _rope_settings(dim: int, theta: float, max_pos: int) -> RopeSettings:
    return RopeSettings(
        head_dim = dim,
        rope_theta = theta,
        rotary_dim = dim,
        max_position_embeddings = max_pos,
        rope_style = RopeStyle.GPTJ,
    )


class HyV4CompressedIndexer(Module):
    """Grouped compressed indexer: pool 4-token keys, score groups, recall 511 + own group."""

    def __init__(self, config: Config, key: str, hidden_size: int, q_lora_rank: int,
                 num_heads: int, head_dim: int, rope_dim: int, compress_ratio: int,
                 group_topk: int, rms_norm_eps: float, rope_theta: float, max_pos: int):
        super().__init__(config = config, key = key, qmap = None)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.compress_ratio = compress_ratio
        self.group_topk = group_topk
        self.rms_eps = rms_norm_eps
        self.rope = RoPE(device = "cpu", rope_settings = _rope_settings(rope_dim, rope_theta, max_pos))
        self.rope.device = None

        self.query_projection = Linear(
            config = config, key = f"{key}.query_projection", in_features = q_lora_rank,
            out_features = num_heads * head_dim, qmap = None, out_dtype = torch.half,
            trim_padded_out = True,
        )
        self.head_weight_projection = Linear(
            config = config, key = f"{key}.head_weight_projection", in_features = hidden_size,
            out_features = num_heads, qmap = None, out_dtype = torch.float, pad_to = 1,
        )
        self.token_key_projection = Linear(
            config = config, key = f"{key}.token_key_projection", in_features = hidden_size,
            out_features = head_dim, qmap = None, out_dtype = torch.half, trim_padded_out = True,
        )
        self.token_key_norm = RMSNorm(config, f"{key}.token_key_norm", rms_norm_eps, out_dtype = torch.half)
        self.key_norm = RMSNorm(config, f"{key}.key_norm", rms_norm_eps, out_dtype = torch.half)
        for m in (self.query_projection, self.head_weight_projection, self.token_key_projection,
                  self.token_key_norm, self.key_norm):
            self.register_submodule(m)

        self.kv_w = None
        self.gate_w = None
        self.intra_group_ape = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.kv_w = stc.get_tensor(f"{self.key}.key_value_projection.weight", device,
                                   no_defer = True, arena = False).float().contiguous()
        self.gate_w = stc.get_tensor(f"{self.key}.key_gate_projection.weight", device,
                                     no_defer = True, arena = False).float().contiguous()
        self.intra_group_ape = stc.get_tensor(f"{self.key}.intra_group_ape", device,
                                              no_defer = True, arena = False).float().contiguous()
        self.rope.device = device
        self.rope.inv_freq = self.rope.inv_freq.to(device)
        self.rope.cached_sin = self.rope.cached_cos = None
        self.rope.cached_sincos_max = 0

    @override
    def unload(self):
        super().unload()
        self.kv_w = self.gate_w = self.intra_group_ape = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.key_value_projection.weight": self.kv_w.half().contiguous(),
            f"{self.key}.key_gate_projection.weight": self.gate_w.half().contiguous(),
            f"{self.key}.intra_group_ape": self.intra_group_ape.half().contiguous(),
        }

    @override
    def optimizer_targets(self):
        return []

    @override
    def optimizer_targets(self):
        return []

    def token_keys(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        k = self.token_key_projection.forward(x, params)
        return self.token_key_norm.forward(k, params, out_dtype = torch.half)

    def pool_groups(self, token_keys: torch.Tensor) -> torch.Tensor:
        n = token_keys.shape[0]
        groups = n // self.compress_ratio
        if groups == 0:
            return token_keys.new_zeros((0, self.head_dim))
        grouped = token_keys[: groups * self.compress_ratio].view(groups, self.compress_ratio, -1).float()
        values = F.linear(grouped, self.kv_w)
        gates = F.linear(grouped, self.gate_w) + self.intra_group_ape
        pooled = (values * gates.softmax(dim = 1)).sum(dim = 1)
        return self.key_norm.forward(pooled.half(), {}, out_dtype = torch.half)

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        group_keys: torch.Tensor,
        query_positions: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        q = self.query_projection.forward(q_resid, params).view(-1, self.num_heads, self.head_dim)
        q_pass, q_rot = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim = -1)
        q_rot = self.rope.apply_torch(q_rot, None, position_ids = query_positions, in_place = False)
        if isinstance(q_rot, tuple):
            q_rot = q_rot[0]
        q = torch.cat((q_pass, q_rot.half()), dim = -1)

        groups = group_keys.shape[0]
        if groups == 0:
            return query_positions.new_zeros((query_positions.shape[0], 0))
        group_pos = (torch.arange(groups, device = group_keys.device) * self.compress_ratio).unsqueeze(0)
        gp = group_pos.expand(query_positions.shape[0], -1)
        k_pass, k_rot = group_keys.split([self.head_dim - self.rope_dim, self.rope_dim], dim = -1)
        k_rot = self.rope.apply_torch(k_rot.view(1, groups, 1, -1), None, position_ids = group_pos,
                                      in_place = False)
        if isinstance(k_rot, tuple):
            k_rot = k_rot[0]
        k_rot = k_rot.view(groups, -1)
        k = torch.cat((k_pass, k_rot.half()), dim = -1)

        scores = torch.einsum("qhd,gd->qhg", q.float(), k.float()).relu()
        weights = self.head_weight_projection.forward(x, params).float()
        weights = weights * (self.num_heads ** -0.5) * (self.head_dim ** -0.5)
        logits = torch.einsum("qh,qhg->qg", weights, scores)

        q_group = query_positions // self.compress_ratio
        gids = torch.arange(groups, device = logits.device).unsqueeze(0)
        logits = logits.masked_fill(gids >= q_group.unsqueeze(1), float("-inf"))
        complete = min(self.group_topk - 1, groups)
        recalled = logits.topk(complete, dim = -1).indices if complete > 0 else \
            query_positions.new_zeros((query_positions.shape[0], 0))
        return torch.cat((recalled, q_group.unsqueeze(1)), dim = 1)


class HyV4Attention(Module):
    """Gated MLA. Sliding layers use 64 heads and a 512-token window; indexed layers use 32
    heads and recall groups selected by the compressed indexer. Both carry learned sinks."""

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        layer_type: str,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        sliding_window: int,
        indexer: HyV4CompressedIndexer | None,
        index_topk: int,
        rope_theta: float,
        rms_norm_eps: float,
        max_position_embeddings: int,
        qmap: str | None = None,
    ):
        super().__init__(config = config, key = key, qmap = qmap)
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope = qk_nope_head_dim
        self.qk_rope = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk = self.qk_head_dim
        self.num_q_heads = num_heads
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.sliding_window = sliding_window if layer_type == "sliding_attention" else 0
        self.indexer = indexer
        self.index_topk = index_topk
        self.sm_scale = self.qk ** -0.5
        self.rms_eps = rms_norm_eps
        self.out_dtype = torch.float
        self.rope = RoPE(device = "cpu", rope_settings = _rope_settings(qk_rope_head_dim, rope_theta,
                                                                       max_position_embeddings))
        self.rope.device = None

        self.q_a_proj = Linear(config, f"{key}.q_a_proj", hidden_size, q_lora_rank,
                               qmap = qmap, out_dtype = torch.half, trim_padded_out = True)
        self.q_a_layernorm = RMSNorm(config, f"{key}.q_a_layernorm", rms_norm_eps, out_dtype = torch.half)
        self.q_b_proj = Linear(config, f"{key}.q_b_proj", q_lora_rank, num_heads * self.qk,
                               qmap = qmap, out_dtype = torch.half, trim_padded_out = True)
        self.kv_a_proj_with_mqa = Linear(config, f"{key}.kv_a_proj_with_mqa", hidden_size,
                                         kv_lora_rank + qk_rope_head_dim, qmap = qmap,
                                         out_dtype = torch.half, trim_padded_out = True)
        self.kv_a_layernorm = RMSNorm(config, f"{key}.kv_a_layernorm", rms_norm_eps, out_dtype = torch.half)
        self.gate_proj = Linear(config, f"{key}.linear_gate", hidden_size, num_heads * v_head_dim,
                                qmap = qmap, out_dtype = torch.half, trim_padded_out = True)
        self.o_proj = Linear(config, f"{key}.o_proj", num_heads * v_head_dim, hidden_size,
                             qmap = qmap, out_dtype = torch.float, trim_padded_out = True)
        self.post_proj_norm = RMSNorm(config, f"{key}.post_proj_norm", rms_norm_eps, out_dtype = torch.float)
        for m in (self.q_a_proj, self.q_a_layernorm, self.q_b_proj, self.kv_a_proj_with_mqa,
                  self.kv_a_layernorm, self.gate_proj, self.o_proj, self.post_proj_norm):
            self.register_submodule(m)
        if self.indexer is not None:
            self.register_submodule(self.indexer)

        self.sinks = None
        self.w_uk = None
        self.w_uv = None
        self.cache_layers = []
        self.idx_plane_dim = indexer.head_dim if indexer is not None else None
        self.index_head_dim = indexer.head_dim if indexer is not None else 0
        self.caps.update({"kv_cache": True})
        self.q_half_bits = False

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        for cl in self.cache_layers:
            cl.alloc(device)
        stc = self.config.stc
        self.sinks = stc.get_tensor(f"{self.key}.learnable_sink_param", device,
                                    no_defer = True, arena = False).float().contiguous()
        w = stc.get_tensor(f"{self.key}.kv_b_proj.weight", device, no_defer = True, arena = False)
        w = w.half().view(self.num_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
        self.w_uk = w[:, : self.qk_nope, :].contiguous()
        self.w_uv = w[:, self.qk_nope :, :].transpose(1, 2).contiguous()
        self.rope.device = device
        self.rope.inv_freq = self.rope.inv_freq.to(device)
        self.rope.cached_sin = self.rope.cached_cos = None
        self.rope.cached_sincos_max = 0
        if self.indexer is not None:
            self.indexer.rope.device = device

    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.sinks = self.w_uk = self.w_uv = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def get_tensors(self):
        t = {f"{self.key}.learnable_sink_param": self.sinks.contiguous()}
        if self.w_uk is not None:
            w = torch.cat([self.w_uk, self.w_uv.transpose(1, 2)], dim = 1)
            t[f"{self.key}.kv_b_proj.weight"] = w.reshape(
                self.num_heads * (self.qk_nope + self.v_head_dim), self.kv_lora_rank).contiguous()
        return t

    def cache_layer_type(self, default, kwargs: dict):
        from ..cache.mla import CacheLayer_MLA_fp16
        return CacheLayer_MLA_fp16, {}

    def _positions(self, bsz: int, seq: int, cache_seqlens, params: dict, device) -> torch.Tensor:
        if cache_seqlens is not None:
            return (cache_seqlens.to(device).unsqueeze(1) + torch.arange(seq, device = device).unsqueeze(0)).int()
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", device, None)
        if positions is not None:
            if positions.dim() == 2:
                return positions[:, -seq:].int()
            return positions.reshape(bsz, -1)[:, -seq:].int()
        return torch.arange(position, position + seq, device = device).unsqueeze(0).expand(bsz, -1).int()

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        bsz, seq, _ = x.shape
        dev = x.device
        cache = params.get("cache")
        layer = None
        cache_seqlens = None
        block_table = None
        if cache is not None:
            layer = cache if not hasattr(cache, "layers") else \
                cache.layers[self.layer_idx, params.get("layer_instance") or 0]
            cache_seqlens = get_for_device(params, "cache_seqlens", dev)
            block_table = get_for_device(params, "block_table", dev)
        pos = self._positions(bsz, seq, cache_seqlens, params, dev)

        gate = self.gate_proj.forward(x, params).view(bsz, seq, self.num_heads, self.v_head_dim)
        q_resid = self.q_a_proj.forward(x, params)
        q_resid = self.q_a_layernorm.forward(q_resid, params, out_dtype = torch.half)
        q = self.q_b_proj.forward(q_resid, params).view(bsz, seq, self.num_heads, self.qk)
        q_nope, q_pe = q.split([self.qk_nope, self.qk_rope], dim = -1)
        kv = self.kv_a_proj_with_mqa.forward(x, params)
        ckv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope], dim = -1)
        ckv = self.kv_a_layernorm.forward(ckv, params, out_dtype = torch.half)
        q_pe, k_pe = self.rope.apply(q_pe, k_pe.unsqueeze(2), position = 0, position_ids = pos)
        k_pe = k_pe.view(bsz, seq, self.qk_rope)

        idx_keys = None
        if self.indexer is not None:
            idx_keys = self.indexer.token_keys(x, params)

        if layer is not None:
            layer.update_kv_direct(cache_seqlens, block_table, ckv, k_pe, seq)
            if idx_keys is not None:
                layer.update_idx_direct(cache_seqlens, block_table, idx_keys, seq)
        else:
            local_ckv, local_kpe, local_idx = ckv, k_pe, idx_keys

        outs = []
        for b in range(bsz):
            if layer is not None:
                pages = block_table[b]
                n = int(cache_seqlens[b].item()) + seq
                ckv_all = layer.k[pages].reshape(-1, self.kv_lora_rank)[:n]
                kpe_all = layer.v[pages].reshape(-1, self.qk_rope)[:n]
                idx_all = layer.k_idx[pages].reshape(-1, self.indexer.head_dim)[:n] if idx_keys is not None else None
            else:
                n = seq
                ckv_all, kpe_all, idx_all = local_ckv[b], local_kpe[b], (local_idx[b] if local_idx is not None else None)
            o = self._attend_one(x[b], q_nope[b], q_pe[b], gate[b], q_resid[b], ckv_all, kpe_all,
                                 idx_all, pos[b], n)
            outs.append(o)
        o = torch.stack(outs, dim = 0).reshape(bsz, seq, self.num_heads * self.v_head_dim)
        o = self.o_proj.forward(o, params)
        o = self.post_proj_norm.forward(o, params, out_dtype = self.out_dtype)
        return to2_out(o, out_dtype, self.out_dtype)

    def _scores_mask(self, qpos: torch.Tensor, kpos: torch.Tensor, window: int) -> torch.Tensor:
        mask = kpos.unsqueeze(0) <= qpos.unsqueeze(1)
        if window:
            mask = mask & (kpos.unsqueeze(0) >= (qpos.unsqueeze(1) - window + 1))
        return mask

    def _attend_one(self, xb, q_nope, q_pe, gate, q_resid, ckv_all, kpe_all, idx_all, posb, n):
        qpos = posb
        seq = qpos.shape[0]
        dev = xb.device
        ql = torch.einsum("qhd,hdc->qhc", q_nope.half(), self.w_uk)
        qpe = q_pe.half()
        kpe_all = kpe_all.half()
        ckv_all = ckv_all.half()
        sinks = self.sinks.float()
        w = self.sliding_window
        out = torch.empty((seq, self.num_heads, self.v_head_dim), dtype = torch.float, device = dev)

        groups = None
        group_keys = None
        if self.indexer is not None and idx_all is not None:
            group_keys = self.indexer.pool_groups(idx_all)
            groups = group_keys.shape[0]

        step = 256 if (w or n <= self.index_topk or self.indexer is None) else 64
        for a in range(0, seq, step):
            b = min(a + step, seq)
            qa = qpos[a:b]
            ql_b = ql[a:b]
            qp_b = qpe[a:b]
            selmask = None
            if w:
                lo = max(0, min(int(qa.min().item()) - w + 1, n))
                hi = min(int(qa.max().item()) + 1, n)
                idx = torch.arange(lo, hi, device = dev)
            elif n > self.index_topk and self.indexer is not None:
                idx, selmask = self._select_groups(xb[a:b], q_resid[a:b], group_keys, groups, qa, n)
            else:
                idx = torch.arange(0, n, device = dev)
            ck = ckv_all[idx]
            kp = kpe_all[idx]
            scores = torch.einsum("qhc,kc->qhk", ql_b, ck)
            scores = scores + torch.einsum("qhd,kd->qhk", qp_b, kp)
            scores = scores.float() * self.sm_scale
            kpos = idx
            mask = self._scores_mask(qa, kpos, w)
            if selmask is not None:
                mask = mask & selmask
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            sk = sinks.view(1, -1, 1).expand(scores.shape[0], -1, 1)
            scores = torch.cat((scores, sk), dim = -1)
            probs = torch.softmax(scores, dim = -1)[..., :-1].half()
            o_lat = torch.einsum("qhk,kc->qhc", probs, ck)
            out[a:b] = torch.einsum("qhc,hcv->qhv", o_lat, self.w_uv).float()
        return (out * torch.sigmoid(gate.float())).to(torch.half)

    def _select_groups(self, xb, q_resid_b, group_keys, groups, qpos, n):
        if groups == 0:
            return torch.arange(0, n, device = qpos.device), None
        recalled = self.indexer.forward(xb, q_resid_b, group_keys, qpos, {})
        offs = torch.arange(self.indexer.compress_ratio, device = qpos.device)
        q_idx = (recalled.unsqueeze(-1) * self.indexer.compress_ratio + offs).flatten(-2)
        q_idx = q_idx.clamp(max = n - 1)
        uidx = torch.unique(q_idx)
        pos = torch.searchsorted(uidx, q_idx)
        selmask = torch.zeros((q_idx.shape[0], uidx.shape[0]), dtype = torch.bool, device = qpos.device)
        selmask.scatter_(1, pos, True)
        return uidx, selmask


def to2_out(x, out_dtype, default):
    return to2(x, out_dtype, default)


class HyV4MoE(Module):
    def __init__(self, config: Config, key: str, hidden_size: int, moe_intermediate: int,
                 latent_size: int | None, num_experts: int, experts_per_tok: int,
                 routed_scaling: float, swiglu_limit: float, qmap: str | None):
        super().__init__(config = config, key = key, qmap = qmap)
        self.routed = BlockSparseMLP(
            config = config,
            key = f"{key}",
            hidden_size = hidden_size,
            intermediate_size = moe_intermediate,
            num_experts = num_experts,
            num_experts_per_tok = experts_per_tok,
            latent_size = latent_size,
            key_latent_in = "routed_expert_down_proj" if latent_size else None,
            key_latent_out = "routed_expert_up_proj" if latent_size else None,
            key_up = "experts.{expert_idx}.up_proj",
            key_gate = "experts.{expert_idx}.gate_proj",
            key_down = "experts.{expert_idx}.down_proj",
            key_routing_gate = "gate",
            key_e_score_bias = "gate.e_score_correction_bias",
            qmap = qmap,
            interm_dtype = torch.half,
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = swiglu_limit,
            router_type = "dots",
            routed_scaling_factor = routed_scaling,
            n_group = 1,
            topk_group = 1,
            shared_experts = None,
        )
        self.shared = GatedMLP(
            config = config,
            key = f"{key}.shared_experts",
            hidden_size = hidden_size,
            intermediate_size = moe_intermediate,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = qmap,
            out_dtype = torch.float,
            activation_fn = "silu",
        )
        self.combine_norm = RMSNorm(config, f"{key}.combine_output_norm",
                                    config.rms_norm_eps, out_dtype = torch.float)
        for m in (self.routed, self.shared, self.combine_norm):
            self.register_submodule(m)

    @override
    def optimizer_targets(self):
        return [self.routed.optimizer_targets(), self.shared.optimizer_targets()]

    def forward(self, x, params, out_dtype = None):
        routed = self.routed.forward(x, params)
        shared = self.shared.forward(x, params)
        return self.combine_norm.forward(routed + shared, params, out_dtype = torch.float)


class HyV4MLP(Module):
    def __init__(self, config: Config, key: str, hidden_size: int, intermediate_size: int,
                 qmap: str | None):
        super().__init__(config = config, key = key, qmap = qmap)
        self.mlp = GatedMLP(
            config = config, key = key, hidden_size = hidden_size,
            intermediate_size = intermediate_size, key_up = "up_proj", key_gate = "gate_proj",
            key_down = "down_proj", qmap = qmap, out_dtype = torch.float, activation_fn = "silu",
        )
        self.post_norm = RMSNorm(config, f"{key}.post_fc2_norm", config.rms_norm_eps,
                                 out_dtype = torch.float)
        self.register_submodule(self.mlp)
        self.register_submodule(self.post_norm)

    @override
    def optimizer_targets(self):
        return [self.mlp.optimizer_targets()]

    def forward(self, x, params, out_dtype = None):
        y = self.mlp.forward(x, params)
        return self.post_norm.forward(y, params, out_dtype = torch.float)


class HyV4EngramLayer(PLELayer):
    """Engram memory: hashed n-gram rows gated into the iHC stream stack before a block.

    The table itself is an NGramEmbedding (streamed from disk); the projections, per-head norms
    and per-stream gates are the checkpoint's engram_layer parameters.
    """

    def __init__(self, config: Config, key: str, layer_idx: int, hidden_size: int, hc_mult: int,
                 ngram_size: int, n_heads: int, n_embed: int, head_vocab_sizes: list,
                 hash_multipliers: list, pad_id: int, fp8_embedding: bool, rms_norm_eps: float,
                 engram_eps: float, stream_from_disk: bool | None = None):
        super().__init__(
            config = config, key = key, layer_idx = layer_idx, hidden_size = hidden_size,
            hc_mult = hc_mult, ple_embed_dim = (ngram_size - 1) * n_embed,
            ngram_size = ngram_size, heads_per_ngram = n_heads, eos_token_id = pad_id,
            conv_kernel_size = 0, rms_norm_eps = rms_norm_eps, stub = True,
        )
        self.stub = False
        self.fp8_embedding = fp8_embedding
        self.pad_id = pad_id
        self.head_dim = n_embed // n_heads
        self.num_heads = (ngram_size - 1) * n_heads
        self.ple_embed_dim = (ngram_size - 1) * n_embed
        self.caps.update({"recurrent_cache": True, "prefetch_ids": True})
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.layer_state_cls = PLELayerState

        self.ple_embedding = NGramEmbedding(
            config = config,
            key = f"{key}.multi_head_embedding.embedding",
            ngram_size = ngram_size,
            heads_per_ngram = n_heads,
            ple_embed_dim = self.ple_embed_dim,
            eos_token_id = pad_id,
            stream_from_disk = stream_from_disk,
            head_vocab_sizes = head_vocab_sizes,
            layer_multipliers = hash_multipliers,
        )
        self.register_submodule(self.ple_embedding)

        self.value_proj = Linear(config, f"{key}.value_proj", self.ple_embed_dim, hidden_size,
                                 qmap = None, out_dtype = torch.half, trim_padded_out = True)
        self.register_submodule(self.value_proj)
        self.key_projs = []
        self.norm_key = []
        self.norm_query = []
        self.head_norms = []
        for h in range(self.num_heads):
            n = RMSNorm(config, f"{key}.lookup_per_head_norms.{h}", engram_eps, out_dtype = torch.half)
            self.head_norms.append(n)
            self.register_submodule(n)
        for i in range(hc_mult):
            kp = Linear(config, f"{key}.key_projs.{i}", self.ple_embed_dim, hidden_size,
                        qmap = None, out_dtype = torch.half, trim_padded_out = True)
            nk = RMSNorm(config, f"{key}.norm_key.{i}", rms_norm_eps, out_dtype = torch.float)
            nq = RMSNorm(config, f"{key}.norm_query.{i}", rms_norm_eps, out_dtype = torch.float)
            self.key_projs.append(kp)
            self.norm_key.append(nk)
            self.norm_query.append(nq)
            for m in (kp, nk, nq):
                self.register_submodule(m)
        self.unified_rmsnorm = RMSNorm(config, f"{key}.unified_rmsnorm", engram_eps,
                                       unweighted = True, out_dtype = torch.float)
        self.register_submodule(self.unified_rmsnorm)
        self.unified_pre_gamma = None
        self.unified_per_hc_scales = None
        self.lookup_table = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.unified_pre_gamma = stc.get_tensor(f"{self.key}.unified_pre_gamma", device,
                                                no_defer = True, arena = False).float().contiguous()
        self.unified_per_hc_scales = [
            stc.get_tensor(f"{self.key}.unified_per_hc_scales.{i}", device,
                           no_defer = True, arena = False).float().contiguous()
            for i in range(self.hc_mult)
        ]
        self.lookup_table = stc.get_tensor(f"{self.key}.lookup_table", "cpu",
                                           no_defer = True, arena = False).long().contiguous()

    @override
    def unload(self):
        super().unload()
        self.unified_pre_gamma = self.unified_per_hc_scales = self.lookup_table = None

    @override
    def get_tensors(self):
        t = {f"{self.key}.unified_pre_gamma": self.unified_pre_gamma.half().contiguous()}
        for i, s in enumerate(self.unified_per_hc_scales):
            t[f"{self.key}.unified_per_hc_scales.{i}"] = s.half().contiguous()
        t[f"{self.key}.lookup_table"] = self.lookup_table.int().contiguous()
        return t

    @override
    def optimizer_targets(self):
        return []

    def _prepare_ids(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to("cpu", torch.int64)
        t = self.lookup_table
        orig = ids
        ids = t[ids.clamp_max(t.shape[0] - 1)]
        return torch.where(orig < t.shape[0], ids, torch.full_like(ids, self.pad_id))

    @staticmethod
    def _fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
        info = torch.finfo(torch.float8_e4m3fn)
        scale = x.detach().abs().amax(dim = -1, keepdim = True).clamp(min = 1e-12) / info.max
        q = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
        return x + (q - x).detach()

    @override
    def forward_streams(self, streams, token_history, params, conv_state = None):
        bsz, seq = streams.shape[:2]
        emb = self.ple_embedding.forward(token_history, params)
        emb = emb.view(bsz, seq, self.num_heads, self.head_dim)
        if self.fp8_embedding:
            emb = self._fp8_round_trip(emb)
        heads = []
        for h, norm in enumerate(self.head_norms):
            heads.append(norm.forward(emb[..., h, :], params,
                                      out_dtype = torch.half).unsqueeze(2))
        emb = torch.cat(heads, dim = 2).flatten(-2)

        gates = []
        for i in range(self.hc_mult):
            key = self.norm_key[i].forward(self.key_projs[i].forward(emb, params), params,
                                           out_dtype = torch.float)
            query = self.norm_query[i].forward(streams[:, :, i], params, out_dtype = torch.float)
            gate = (key * query).sum(dim = -1) * self.hidden_size ** -0.5
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim = 2).unsqueeze(-1)

        main = self.unified_pre_gamma * self.unified_rmsnorm.forward(
            self.value_proj.forward(emb, params), params, out_dtype = torch.float)
        branch = torch.stack([s * main for s in self.unified_per_hc_scales], dim = 2)
        delta = gates * main.unsqueeze(2) + branch
        return delta.float(), None
