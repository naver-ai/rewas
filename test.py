# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import os

from audioldm.pipeline import rewas_generation, build_control_model

import json
import argparse
import pandas as pd
from random import shuffle
from omegaconf import OmegaConf
from tqdm import tqdm 

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

from utils import seed_everything
from encoder.encoder_utils import patch_config, get_pretrained
from encoder.phi import Phi


def main(args):

    seed = args.seed
    seed_everything(seed)

    assert os.path.isfile(args.ckpt_path), "check checkpoints in ckpt_path!"
        
    control_type = args.control_type
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    audioldm_control = build_control_model(
        ckpt_path = args.ckpt_path,
        control_type = args.control_type,
        config = args.config,
        model_name = args.model_name)

    cfg_path = f'./configs/cfg-{args.synchformer_exp}.yaml'
    synchformer_cfg = OmegaConf.load(cfg_path)
    synchformer_cfg = patch_config(synchformer_cfg)
    

    video_encoder = get_pretrained(args.synchformer_exp, 0)

    phi = Phi()
    resume_params = torch.load(args.phi_ckpt_path)
    resume_new = {k.replace("module.",""): v for k, v in resume_params.items()}
    phi.load_state_dict(resume_new)

    phi.eval()
    phi = nn.DataParallel(phi, device_ids=[i for i in range(torch.cuda.device_count())])

    print(f'Generate data list: {args.testlist}')

    with open(args.testlist, 'rb') as f:
        datalist = list(map(json.loads, f))


    for x in tqdm(datalist):
        prompt = x['prompt']
        videopath = x['video_name']

        waveform = rewas_generation(
            audioldm_control,
            prompt,
            videopath,
            args.control_type,
            args.synchformer_exp,
            synchformer_cfg,
            video_encoder,
            phi,
            args.file_path,
            seed,
            duration=args.duration,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            n_candidate_gen_per_text=args.n_candidate_gen_per_cond,
            batchsize=args.batchsize,
            save_path=save_path,
            re_encode=args.re_encode,
            local_rank=0
            )

    if args.re_encode:
        os.rmdir('.cache/')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--testlist",
        type=str,
        default="test_samples.json",
    )

    parser.add_argument(
        "--datadir",
        type=str,
        default="/path/to/video",
    )

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        help="The path to save model output",
        default="./results",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="The checkpoint you gonna use",
        default="audioldm-m-full",
    )

    parser.add_argument(
        "-ckpt",
        "--ckpt_path",
        type=str,
        help="The path to the pretrained .ckpt model",
        default="ckpts/audioldm_m_rewas_vggsound.ckpt",
    )
    
    parser.add_argument(
        "--phi_ckpt_path",
        type=str,
        help="The path to the pretrained .ckpt video encoder",
        default="ckpts/phi_vggsound.ckpt",
    )
    
    parser.add_argument(
        "--synchformer_exp",
        type=str,
        help="The name of experiment of synchformer",
        default="24-01-04T16-39-21",
    )

    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=1,
        help="Generate how many samples at the same time",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="The sampling step for DDIM",
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        default=3,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
    )

    parser.add_argument(
        "-dur",
        "--duration",
        type=float,
        default=5.0,
        help="The duration of the samples",
    )

    parser.add_argument(
        "-n",
        "--n_candidate_gen_per_cond",
        type=int,
        default=1,
        help="The number of generated sample per condition. A Larger value usually lead to better quality with heavier computation",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/audioldm_m_rewas.yaml",
    )


    parser.add_argument(
        "--control_type",
        type=str,
        default="energy_video",
        choices=["energy_audio", "energy_video"]
    )

    parser.add_argument('--re_encode', action='store_true')


    args = parser.parse_args()

    main(args)


