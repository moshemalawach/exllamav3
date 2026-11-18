from __future__ import annotations
from typing_extensions import override
import os
import numpy as np
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm, Embedding, Linear, ExpandStreams, TransformerBlock,
)
from ..modules.attn import prepare_for_attn
from ..modules.hyv4 import (
    HyV4Attention, HyV4CompressedIndexer, HyV4EngramLayer, HyV4HyperConnection,
    HyV4HyperHead, HyV4MLP, HyV4MoE,
)

# Hy4 flash (Tencent hy_v4_flash): 42 layers, 1 dense + 41 LatentMoE, hybrid 12 indexed (DSA)
# + 30 sliding-window MLA layers, identity hyper-connections (4 streams) and Engram memory at
# layers 1/11/21/31. Reference: transformers hy_v4_flash + hy_v4, vLLM hy_v4_flash PR.
# Text-only: the vision tower and DFly2 drafter stay in the checkpoint but are not loaded.

_RATIO_THETA_SLIDING = 10000.0
_RATIO_THETA_INDEXED = 10000000.0


def _engram_multipliers(seed: int, layer_id: int, compressed_vocab: int, max_ngram: int):
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    rng = np.random.default_rng(seed + 10007 * layer_id)
    drawn = rng.integers(0, half_bound, size = max_ngram, dtype = np.int64)
    return (drawn * 2 + 1).tolist()


class HyV4FlashConfig(Config):
    arch_string = "HYV4FlashForCausalLM"
    cfg_prefix = "text_config->"

    def __init__(self, directory: str, model_classes: dict | None = None, **kwargs):
        super().__init__(directory, model_classes or {"text": HyV4FlashModel}, **kwargs)
        p = self.cfg_prefix

        self.vocab_size = self.read_cfg(int, p + "vocab_size", no_default)
        self.hidden_size = self.read_cfg(int, p + "hidden_size", no_default)
        self.intermediate_size = self.read_cfg(int, p + "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, p + "moe_intermediate_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, p + "num_hidden_layers", no_default)
        self.num_q_heads = self.read_cfg(int, p + "num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, p + "q_lora_rank", no_default)
        self.kv_lora_rank = self.read_cfg(int, p + "kv_lora_rank", no_default)
        self.qk_nope = self.read_cfg(int, p + "qk_nope_head_dim", no_default)
        self.qk_rope = self.read_cfg(int, p + "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, p + "v_head_dim", no_default)
        self.num_experts = self.read_cfg(int, p + "n_routed_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, p + "num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, p + "n_shared_experts", 1)
        self.routed_scaling_factor = self.read_cfg(float, p + "routed_scaling_factor", 1.0)
        self.routed_expert_hidden_size = self.read_cfg(int, p + "routed_expert_hidden_size", None)
        self.swiglu_limit = self.read_cfg(float, p + "swiglu_limit", 0.0)
        self.rms_norm_eps = self.read_cfg(float, p + "rms_norm_eps", 1e-5)
        self.sliding_window = self.read_cfg(int, p + "sliding_window", 512)
        self.swa_num_attention_heads = self.read_cfg(int, p + "swa_num_attention_heads", 64)
        self.swa_qk_nope = self.read_cfg(int, p + "swa_qk_nope_head_dim", self.qk_nope)
        self.swa_qk_rope = self.read_cfg(int, p + "swa_qk_rope_head_dim", self.qk_rope)
        self.swa_v_head_dim = self.read_cfg(int, p + "swa_v_head_dim", self.v_head_dim)
        self.indexer_compress_ratio = self.read_cfg(int, p + "compressed_indexer_compress_ratio", 4)
        self.indexer_num_heads = self.read_cfg(int, p + "compressed_indexer_num_attention_heads", 16)
        self.indexer_head_dim = self.read_cfg(int, p + "compressed_indexer_head_dim", 128)
        self.indexer_group_topk = self.read_cfg(int, p + "compressed_indexer_group_topk", 512)
        self.index_topk = self.indexer_group_topk * self.indexer_compress_ratio
        self.layer_types = self.read_cfg(list, p + "layer_types", no_default)
        self.mlp_layer_types = self.read_cfg(list, p + "mlp_layer_types", no_default)
        self.max_position_embeddings = self.read_cfg(int, p + "max_position_embeddings", 1048576)
        self.tie_word_embeddings = self.read_cfg(bool, p + "tie_word_embeddings", False)

        self.hc_mult = self.read_cfg(int, p + "hc_mult", 4)
        self.hc_magnitude = self.read_cfg(float, p + "hc_magnitude", 2.0)
        self.hc_eps = self.read_cfg(float, p + "hc_eps", 1e-6)

        self.engram_layer_ids = self.read_cfg(list, p + "engram_layer_ids", [])
        self.engram_max_ngram_size = self.read_cfg(int, p + "engram_max_ngram_size", 3)
        self.engram_n_head_per_ngram = self.read_cfg(int, p + "engram_n_head_per_ngram", 8)
        self.engram_n_embed_per_ngram = self.read_cfg(int, p + "engram_n_embed_per_ngram", 768)
        self.engram_layer_head_vocab_sizes = self.read_cfg(dict, p + "engram_layer_head_vocab_sizes", {})
        self.engram_pad_id = self.read_cfg(int, p + "engram_pad_id", 2)
        self.engram_fp8_embedding = self.read_cfg(bool, p + "engram_fp8_embedding", True)
        self.engram_seed = self.read_cfg(int, p + "engram_seed", 0)
        self.engram_lookup_table_size = self.read_cfg(int, p + "engram_lookup_table_size", 120818)
        self.use_engram = os.environ.get("HYV4_NO_ENGRAM", "0") != "1"

        self.head_dim = self.qk_nope + self.qk_rope

    def attn_dims(self, idx: int):
        sliding = self.layer_types[idx] == "sliding_attention"
        if sliding:
            return (self.swa_num_attention_heads, self.swa_qk_nope, self.swa_qk_rope,
                    self.swa_v_head_dim, _RATIO_THETA_SLIDING)
        return (self.num_q_heads, self.qk_nope, self.qk_rope, self.v_head_dim,
                _RATIO_THETA_INDEXED)


class HyV4FlashModel(Model):
    config_class = HyV4FlashConfig

    def __init__(self, config: HyV4FlashConfig, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        self.caps.update({"supports_tp": False})

        self.modules += [
            Embedding(config = config, key = f"{key_prefix}.embed_tokens",
                      vocab_size = config.vocab_size, hidden_size = config.hidden_size),
            ExpandStreams(config = config, key = "hc_expand", hc_mult = config.hc_mult),
        ]
        self.first_block_idx = len(self.modules)

        engram_meta = {}
        if config.use_engram and config.engram_layer_ids:
            for layer_id in config.engram_layer_ids:
                key = f"{key_prefix}.layers.{layer_id - 1}.engram_layer"
                lookup = config.stc.get_tensor(f"{key}.lookup_table", "cpu", no_defer = True)
                lookup = lookup.long()
                compressed = int(lookup.max()) + 1
                sizes = [s for order in config.engram_layer_head_vocab_sizes[str(layer_id)]
                         for s in order]
                mults = _engram_multipliers(config.engram_seed, layer_id, compressed,
                                            config.engram_max_ngram_size)
                engram_meta[layer_id] = (lookup, sizes, mults, int(lookup[config.engram_pad_id]))

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"
            if idx + 1 in engram_meta:
                lookup, sizes, mults, pad = engram_meta[idx + 1]
                self.modules.append(HyV4EngramLayer(
                    config = config, key = f"{key}.engram_layer", layer_idx = -(idx + 1),
                    hidden_size = config.hidden_size, hc_mult = config.hc_mult,
                    ngram_size = config.engram_max_ngram_size,
                    n_heads = config.engram_n_head_per_ngram,
                    n_embed = config.engram_n_embed_per_ngram,
                    head_vocab_sizes = sizes, hash_multipliers = mults, pad_id = pad,
                    fp8_embedding = config.engram_fp8_embedding,
                    rms_norm_eps = config.rms_norm_eps,
                    engram_eps = torch.finfo(torch.float32 if "float32" in str(config.config_dict.get("dtype", "bfloat16")) else torch.bfloat16).eps,
                ))
                self.modules[-1].lookup_table = lookup
                self.modules[-1].ple_embedding.eos_token_id = pad

            n_heads, nope, rope, vdim, theta = config.attn_dims(idx)
            indexer = None
            if config.layer_types[idx] != "sliding_attention":
                indexer = HyV4CompressedIndexer(
                    config = config, key = f"{key}.self_attn.compressed_indexer",
                    hidden_size = config.hidden_size, q_lora_rank = config.q_lora_rank,
                    num_heads = config.indexer_num_heads, head_dim = config.indexer_head_dim,
                    rope_dim = config.qk_rope, compress_ratio = config.indexer_compress_ratio,
                    group_topk = config.indexer_group_topk, rms_norm_eps = config.rms_norm_eps,
                    rope_theta = _RATIO_THETA_INDEXED,
                    max_pos = config.max_position_embeddings,
                )
            attn = HyV4Attention(
                config = config, key = f"{key}.self_attn", layer_idx = idx,
                layer_type = config.layer_types[idx], hidden_size = config.hidden_size,
                num_heads = n_heads, qk_nope_head_dim = nope, qk_rope_head_dim = rope,
                v_head_dim = vdim, q_lora_rank = config.q_lora_rank,
                kv_lora_rank = config.kv_lora_rank, sliding_window = config.sliding_window,
                indexer = indexer, index_topk = config.index_topk, rope_theta = theta,
                rms_norm_eps = config.rms_norm_eps,
                max_position_embeddings = config.max_position_embeddings,
                qmap = "block.attn",
            )
            if config.mlp_layer_types[idx] == "sparse":
                mlp = HyV4MoE(
                    config = config, key = f"{key}.mlp", hidden_size = config.hidden_size,
                    moe_intermediate = config.moe_intermediate_size,
                    latent_size = config.routed_expert_hidden_size,
                    num_experts = config.num_experts,
                    experts_per_tok = config.num_experts_per_tok,
                    routed_scaling = config.routed_scaling_factor,
                    swiglu_limit = config.swiglu_limit, qmap = "block.mlp",
                )
            else:
                mlp = HyV4MLP(config = config, key = f"{key}.mlp", hidden_size = config.hidden_size,
                              intermediate_size = config.intermediate_size, qmap = "block.mlp")
            self.modules.append(TransformerBlock(
                config = config, key = key, layer_idx = idx,
                attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
                attn = attn,
                attn_hc = HyV4HyperConnection(
                    config = config, key = f"{key}.hc_attn_layer.hc_pre",
                    hc_mult = config.hc_mult, hidden_size = config.hidden_size,
                    magnitude = config.hc_magnitude, eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps),
                mlp_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
                mlp = mlp,
                mlp_hc = HyV4HyperConnection(
                    config = config, key = f"{key}.hc_mlp_layer.hc_pre",
                    hc_mult = config.hc_mult, hidden_size = config.hidden_size,
                    magnitude = config.hc_magnitude, eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps),
            ))

        self.last_kv_module_idx = len(self.modules) - 1

        head_key = None
        if config.tie_word_embeddings and not config.stc.has_tensor("lm_head.weight"):
            head_key = f"{key_prefix}.embed_tokens"
        self.modules += [
            HyV4HyperHead(config = config, key = f"{key_prefix}.hc_head",
                          hc_mult = config.hc_mult, hidden_size = config.hidden_size,
                          eps = config.hc_eps, rms_norm_eps = config.rms_norm_eps),
            RMSNorm(config, f"{key_prefix}.norm", config.rms_norm_eps, out_dtype = torch.half),
            Linear(config, "lm_head", config.hidden_size, config.vocab_size,
                   alt_key = head_key, qmap = "head", caps = {"logits_output": True}),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        from ..cache.dsa import DSV4State
        self.recurrent_state_cls = DSV4State
        self.caps.update({"recurrent_states": True})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # HY4 chat template, no thinking header
        start, mid, end = "<｜hy_start:opensource｜>", "<｜hy_middle:opensource｜>", "<｜hy_end:opensource｜>"
        p = ""
        if system_prompt:
            p += f"{start}system{mid}{system_prompt}{end}"
        p += f"{start}user{mid}{prompt}{end}"
        p += f"{start}assistant{mid}"
        return p
