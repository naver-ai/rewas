# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import argparse
import time
import numpy as np
import einops
import os
import torch
from torch.utils.data import DataLoader
from audioldm.utilities.data.dataset import AudioDataset
import yaml
from tqdm import tqdm


def get_audio(audio):
    audio = torch.mean(audio, axis=1)
    return audio

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def filter_common_keys(dict_a, dict_b):
    # Find the common keys between dict_a and dict_b
    common_keys = dict_a.keys() & dict_b.keys()
    
    # Create new dictionaries with only the common keys
    filtered_dict_a = {key: dict_a[key] for key in common_keys}
    filtered_dict_b = {key: dict_b[key] for key in common_keys}
    sorted_dict_a = dict(sorted(filtered_dict_a.items()))
    sorted_dict_b = dict(sorted(filtered_dict_b.items()))
    return sorted_dict_a, sorted_dict_b



def main(args):
    batch_size = args.batch_size
    configs = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    
    generated_dataset = AudioDataset(configs, split="video_control", add_ons=dataloader_add_ons)
    generated_dataloader = DataLoader(generated_dataset, num_workers=8, batch_size=batch_size, shuffle=True,drop_last =True)
    
    gtdataset = AudioDataset(configs, split="gt", add_ons=dataloader_add_ons)
    gt_dataloader = DataLoader(gtdataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True,drop_last =True)
    
    gt_energy = {}
    test_energy = {}

    for idx, item in tqdm(enumerate(gt_dataloader)):
        name = str(item['fname'][0]).split("/")[-1]
        gt_mel = item['log_mel_spec']
        energy = torch.mean(gt_mel, dim=2)
        gt_energy[f'{name}'] = energy

    for idx, item in tqdm(enumerate(generated_dataloader)):
        name = str(item['fname'][0]).split("/")[-1]
        pred_mel = item['log_mel_spec']
        energy = torch.mean(pred_mel, dim=2)
        test_energy[f'{name}'] = energy

    gt_energy, test_energy = filter_common_keys(gt_energy, test_energy)
    print(gt_energy.keys())
    print(test_energy.keys())
    gt_energy = torch.cat(list(gt_energy.values()),dim=0)
    test_energy = torch.cat(list(test_energy.values()),dim=0)
    loss = nn.L1Loss()
    MAE  = loss(gt_energy, test_energy)
    print(len(gt_energy.keys()))
    print(f"###### MAE: {MAE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--config', default='configs/audioldm_m_rewas.yaml', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_path', default="outputs", type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    args = parser.parse_args()
    main(args)
