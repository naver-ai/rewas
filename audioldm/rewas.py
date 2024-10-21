# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import einops
import torch
import torch as th
import torch.nn as nn
import os
import numpy as np

from encoder.encoder_utils import instantiate_from_config
from audioldm.utilities.data.utils import save_wave
from audioldm.utilities.diffusion_util import (
    conv_nd,
    linear,
    normalization,
    zero_module,
    timestep_embedding
)

from audioldm.latent_diffusion.ddpm import LatentDiffusion
from audioldm.latent_diffusion.ddim import DDIMSampler
from audioldm.diffusionmodules.attention import SpatialTransformer
from audioldm.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, Upsample
from einops.layers.torch import Rearrange, Reduce



class ReWaS(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, cond_dict = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = control.to(memory_format=torch.contiguous_format).float()
        control = control.squeeze()
        
        if control.shape[0] == 512:
            control = control.unsqueeze(0)
        control = einops.rearrange(control, 'b w -> b 1 1 w')
        control = einops.repeat(control,'b 1 1 w -> b 1 w h', h=64)
        
        encoder_posterior = self.encode_first_stage(control)
        control = self.get_first_stage_encoding(encoder_posterior).detach()
        cond_dict["c_concat"] = [control]

        return x, cond_dict

    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name=None,
        use_plms=False,
        limit_num=None,
        save_path=None,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        if name is None:
            name = self.get_validation_folder_name()
        
        try:
            waveform_save_path = os.path.join(self.get_log_dir(), name)
            waveform_save_path = waveform_save_path.replace("val_0", "infer")
        except:
            waveform_save_path = save_path
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform inference save path: ", waveform_save_path)

        with self.ema_scope("Plotting"):
            for i, batch in enumerate(batchs):
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    unconditional_prob_cfg=0.0,  # Do not output unconditional information in the c
                )

                if limit_num is not None and i * z.size(0) > limit_num:
                    break
                
                text = super(LatentDiffusion, self).get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen

                # Generate multiple samples at a time and filter out the best
                # The condition to the diffusion wrapper can have many format
                for cond_key in c.keys():
                    if isinstance(c[cond_key], list):
                        for i in range(len(c[cond_key])):
                            c[cond_key][i] = torch.cat([c[cond_key][i]] * n_gen, dim=0)
                    elif isinstance(c[cond_key], dict):
                        for k in c[cond_key].keys():
                            c[cond_key][k] = torch.cat([c[cond_key][k]] * n_gen, dim=0)
                    else:
                        c[cond_key] = torch.cat([c[cond_key]] * n_gen, dim=0)
                
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = {}
                    for key in self.cond_stage_model_metadata:
                        model_idx = self.cond_stage_model_metadata[key]["model_idx"]
                        unconditional_conditioning[key] = self.cond_stage_models[
                            model_idx
                        ].get_unconditional_condition(batch_size)
                # unconditional control
                if not c.get("c_concat") == None:
                    unconditional_conditioning["c_concat"] = [torch.zeros_like(c["c_concat"][0])]
                
                fnames = list(super(LatentDiffusion, self).get_input(batch, "fname"))

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )  

                new_names = []
                for fname in fnames:
                    new_names += [os.path.split(fname)[-1][:-4]]*n_gen

                mel = self.decode_first_stage(samples)
                waveform = self.mel_spectrogram_to_waveform(mel)

                out_paths = save_wave(waveform, waveform_save_path, name=["generated_"+ x for x in new_names]) # real_name
        
        return waveform, out_paths


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = cond["film_clap_cond1"].squeeze(1)
        
        assert isinstance(cond['c_concat'],list)
        
        cond_control = torch.cat(cond['c_concat'], 1)

        if cond_control is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context_list = [],y = cond_txt, control=None, only_mid_control=self.only_mid_control, context_attn_mask_list=[])
        else:
            control = self.control_model(x=x_noisy, hint=cond_control, timesteps=t, y = cond_txt, context_list = [], context_attn_mask_list=[])
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context_list = [],y = cond_txt, control=control, only_mid_control=self.only_mid_control, context_attn_mask_list=[])

        if isinstance(eps, tuple) and not return_ids:
            return eps[0]

        return eps


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x, c = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):
        if mask is not None:
            shape = (self.channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.latent_t_size, self.latent_f_size)

        intermediate = None
        if ddim and not use_plms:
            print("Use ddim sampler")

            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        elif use_plms:
            print("Use plms sampler")
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            print("Use DDPM sampler")
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


# reference: https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py

class Adapter(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            extra_sa_layer=True,
            num_classes=None,
            extra_film_condition_dim=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=True,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"


        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        # self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.extra_film_condition_dim = extra_film_condition_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.dims = dims
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        # assert not (
        #     self.num_classes is not None and self.extra_film_condition_dim is not None
        # ), "As for the condition of theh UNet model, you can only set using class label or an extra embedding vector (such as from CLAP). You cannot set both num_classes and extra_film_condition_dim."

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.use_extra_film_by_concat = self.extra_film_condition_dim is not None

        if self.extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)
            print(
                "+ Use extra condition on UNet channel using Film. Extra condition dimension is %s. "
                % self.extra_film_condition_dim
            )

        if context_dim is not None and not use_spatial_transformer:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        elif context_dim is None:
            context_dim = [None]  # At least use one spatial transformer

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if extra_sa_layer:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=None,
                            )
                        )
                    for context_dim_id in range(len(context_dim)):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim[context_dim_id],
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        middle_layers = [
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if extra_sa_layer:
            middle_layers.append(
                SpatialTransformer(
                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=None
                )
            )
        for context_dim_id in range(len(context_dim)):
            middle_layers.append(
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim[context_dim_id],
                )
            )
        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        self._feature_size += ch
        self.middle_block_out = self.make_zero_conv(ch)

        self.input_hint_block = TimestepEmbedSequential(
            Rearrange('b c w h -> (b c w) h'),
            nn.Linear(16,16),
            nn.SiLU(),
            Rearrange('(b c w) h -> b c w h', c=8, w=128, h=16),
            zero_module(conv_nd(dims, 8, model_channels, 3, padding=1))
        )
        

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(
        self,
        x,
        hint,
        timesteps,
        y,
        context_list=None,
        context_attn_mask_list=None,
        **kwargs,
    ):

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.use_extra_film_by_concat: # yes
            emb = th.cat([emb, self.film_emb(y)], dim=-1)

        guided_hint = self.input_hint_block(hint, emb, context_list, context_attn_mask_list) # 여기서 오류남 # torch.Size([24, 192, 128, 16]) 

        outs = []
        
        h = x.type(self.dtype) # torch.Size([24, 8, 128, 16])
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context_list, context_attn_mask_list)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context_list, context_attn_mask_list) # 여기서 오류 나는 것 같은데 
            outs.append(zero_conv(h, emb, context_list, context_attn_mask_list))

        h = self.middle_block(h, emb, context_list, context_attn_mask_list)
        outs.append(self.middle_block_out(h, emb, context_list, context_attn_mask_list))

        return outs


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context_list=None, control=None, only_mid_control=False, y=None, context_attn_mask_list=None, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            
            if self.use_extra_film_by_concat:
                emb = th.cat([emb, self.film_emb(y)], dim=-1)
            
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context_list, context_attn_mask_list)
                hs.append(h)
            h = self.middle_block(h, emb, context_list, context_attn_mask_list)
        
    
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context_list, context_attn_mask_list)

        h = h.type(x.dtype)
        return self.out(h)
