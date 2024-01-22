from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel

from nnsight.util import WrapperModule

from ..patching import Patch, Patcher
from .LanguageModel import LanguageModel


class Mamba(LanguageModel):
    def _load_meta(
        self, repoid_or_path, *args, device=None, **kwargs
    ) -> PreTrainedModel:
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config_data = load_config_hf(repoid_or_path)
        self.config = MambaConfig(**config_data)
        return MambaLMHeadModel(self.config, device="meta", dtype=None, **kwargs)

    def _load_local(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        model = MambaLMHeadModel(self.config, **kwargs)
        model.load_state_dict(load_state_dict_hf(repoid_or_path, **kwargs))
        return model

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return BatchEncoding({"input_ids": torch.tensor([[0]])})

    def _scan(self, prepared_inputs, *args, **kwargs) -> None:
        def blah(hs, *args, residual=None, **kwargs):
            return hs, residual

        def blah1(hs, *args, **kwargs):
            return hs

        def blah2(hs, *args, **kwargs):
            return hs

        def blah3(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus):
            return (
                conv1d_out,
                torch.zeros((*conv1d_out.shape, A.shape[1] * 2), device="meta"),
                conv1d_out,
            )

        with Patcher() as patcher:
            patcher.add(Patch(mamba_ssm.modules.mamba_simple, blah, "rms_norm_fn"))
            patcher.add(Patch(mamba_ssm.models.mixer_seq_simple, blah1, "rms_norm_fn"))
            patcher.add(Patch(causal_conv1d_cuda, blah2, "causal_conv1d_fwd"))
            patcher.add(Patch(selective_scan_cuda, blah3, "fwd"))

            self.meta_model(prepared_inputs.copy()["input_ids"].to("meta"))

    def _forward(self, prepared_inputs, *args, **kwargs) -> Any:
        return self.local_model(
            prepared_inputs["input_ids"].to(next(self.local_model.parameters()).device),
            *args,
            **kwargs,
        )

    def _generation(self, prepared_inputs, *args, max_length: int = 1, **kwargs) -> Any:
        return self.local_model.generate(
            prepared_inputs["input_ids"].to(next(self.local_model.parameters()).device),
            *args,
            max_length=max_length,
            **kwargs,
        )


class MambaModuleInterp(mamba_ssm.modules.mamba_simple.Mamba):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dt = WrapperModule()
        self.B = WrapperModule()
        self.C = WrapperModule()

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        
        xz = rearrange(
            self.in_proj(hidden_states),
            "b l d -> b d l",
            l=seqlen,
        )
        

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(
                F.pad(x, (self.d_conv - x.shape[-1], 0))
            )  # Update state (B D W)
            
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt).t()

        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        dt = self.dt(dt)
        B = self.B(B)
        C = self.C(C)

        y = mamba_ssm.ops.selective_scan_interface.selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out


class MambaInterp(Mamba):
    def __init__(self, *args, **kwargs):
    
        patcher = Patcher()

        patcher.add(
            Patch(
                mamba_ssm.models.mixer_seq_simple,
                MambaModuleInterp,
                "Mamba",
            )
        )

        patcher.__enter__()

        super().__init__(*args, **kwargs)
