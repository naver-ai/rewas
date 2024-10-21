# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import yaml
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import savgol_filter

from utils import seed_everything, default_audioldm_config
from encoder.encoder_utils import get_transforms, get_pretrained
from audioldm.rewas import ReWaS
from audioldm.utilities.audio import read_wav_file
from audioldm.utilities.model_util import set_cond_audio, set_cond_text
from audioldm.utilities.data.utils import re_encode_video, get_video_and_audio, save_video


def make_batch_for_control_to_audio(
    text, videos, control_type, synchformer_cfg,  
    waveform=None, fbank=None, batchsize=1, local_rank = None, re_encode=True):
    
    if not type(text)==list:
        text = [text] * batchsize
        video = [videos] * batchsize

    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
        
    model_inputs = []
    fbank_list = []

    if waveform is None:
        waveform = torch.zeros((batchsize, 160000)) 
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    if fbank is None:
        fbank = torch.zeros((batchsize, 1024, 64)) 
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize    
    
    fbank_list.append(fbank)
    
    for idx, video in enumerate(videos):
        
        x = get_input(
            video,
            synchformer_cfg,
            control_type=control_type,
            local_rank=local_rank,
            re_encode=re_encode)

        model_inputs.append(x)
            
    fbank = torch.cat(fbank_list,dim=0)
    model_inputs = torch.cat(model_inputs,dim=0)

    batch = {
            "fname": videos,  # list
            "text": text,  # list
            "label_vector": None,
            "waveform": waveform,
            "model_inputs": model_inputs,
            # "energy":control,
            "log_mel_spec": fbank,
            "stft": torch.zeros((batchsize, 1024, 512)) # Not used in inference

    }
    return batch

def rewas_generation(
        latent_diffusion,
        text,
        videos,
        control_type,
        synchformer_exp,
        synchformer_cfg,
        video_encoder,
        phi,
        original_audio_file_path = None,
        seed=42,
        ddim_steps=200,
        duration=5,
        batchsize=1,
        guidance_scale=2.5,
        n_candidate_gen_per_text=1,
        re_encode=True,
        config=None,
        save_path=None,
        local_rank=None
    ):
    
    seed_everything(int(seed))
    
    waveform = None
    
    if original_audio_file_path is not None:
        print(f'{original_audio_file_path=}')
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)

    if type(text)==list:
        batchsize = len(text)

    if not type(videos)==list:
        videos = [videos]

    batch = make_batch_for_control_to_audio(
        text, videos, control_type, synchformer_cfg,
        waveform=waveform, batchsize=batchsize, local_rank=local_rank, re_encode=re_encode)

    with torch.set_grad_enabled(False):
        with torch.autocast('cuda', enabled=synchformer_cfg.training.use_half_precision):
            hint = batch['model_inputs'].to(f'cuda:{local_rank}')
            hint = hint.permute(0, 1, 3, 2, 4, 5) # (B, S, C, Tv, H, W)
            hint = video_encoder(hint, for_loop=False)[0] # if for_loop = True: Segment is not combined with batch
            B, S, tv, D = hint.shape
            hint = hint.view(B, S*tv, D)

        hint = phi(hint.float())
        hint = hint.squeeze(2)
        
        control = savgol_filter(hint.cpu(), 9, 2)
        control = torch.tensor(control)
        control = F.interpolate(control.unsqueeze(1), size=512)
        control = torch.tensor(control).squeeze(1)

    batch["energy"] = control
    
    latent_diffusion.latent_t_size = int(duration * 25.6) # duration to latent size
    
    if waveform is not None:
        print(f"Generate audio that has similar content as {original_audio_file_path}")
        latent_diffusion = set_cond_audio(latent_diffusion)
    else:
        latent_diffusion = set_cond_text(latent_diffusion)


    with torch.no_grad():
        waveform, waveform_save_path = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_candidate_gen_per_text,
            duration=duration,
            save_path = save_path,
            control_plt = True
        )

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    if n_candidate_gen_per_text > 1:
        videos = videos * n_candidate_gen_per_text

    args = [(path, vid) for path, vid in zip(waveform_save_path, videos)]
    pool.starmap(save_video, args)

    pool.close()
    pool.join()


def build_control_model(
    control_type,
    ckpt_path=None,
    config=None,
    model_name="audioldm-m-full",
    distribution=False,
    device = "cuda:0",
    local_rank = None
):
    print(f"Load AudioLDM: {model_name}")
    
    control  = False
    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        control  = True
    else:
        config = default_audioldm_config(model_name)
        
    # Use text as condition instead of using waveform during training
    config["model"]["params"]["cond_stage_key"] = "text"
    
    latent_diffusion = ReWaS(**config["model"]["params"])

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)


    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    if distribution: 
        latent_diffusion = torch.nn.parallel.DistributedDataParallel(latent_diffusion, device_ids=[local_rank])
        dist.barrier() 
        latent_diffusion = latent_diffusion.module

    model_idx = latent_diffusion.cond_stage_model_metadata['film_clap_cond1']["model_idx"]
    latent_diffusion.cond_stage_models[model_idx].embed_mode = "text"
    latent_diffusion.control_key = "energy"

    return latent_diffusion

def get_input(
        video_path,
        synchformer_cfg,
        control_type,
        device: str = 'cuda',
        audio_sr: int = 16000,
        video_fps: int = 25, 
        in_size: int = 256,
        duration: int = 5,
        local_rank = None,
        re_encode=True
        ):
    
    if video_path is None:
        return None

    if control_type == "energy_video":

        if re_encode:
            video_path = re_encode_video('.cache', video_path, video_fps, audio_sr, in_size)
        
        rgb, audio, meta = get_video_and_audio(
            video_path, get_meta=True, duration=duration, start_sec=0, end_sec=None, random_start=0)

        item_temp = dict(
                video=rgb, audio=audio, meta=meta, path=video_path, split='test',
                targets={'v_start_i_sec': 0.0, 'offset_sec': 0.0 },
            )

        item_temp = get_transforms(synchformer_cfg, ['test'])['test'](item_temp) 
        
        x = item_temp['video'].unsqueeze(0)

    else:
        print('Undefined control type')

    return x