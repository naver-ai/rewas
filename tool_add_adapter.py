# reference: https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control.py

import sys
import os
import argparse
import torch
from omegaconf import OmegaConf
from encoder.encoder_utils import instantiate_from_config


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def main():
    parser = argparse.ArgumentParser(description="Adding adapter architecture to audioldm.")
    parser.add_argument("--input_path", type=str, required=True, help="path to pretrained model weights")
    parser.add_argument("--output_path", type=str, required=True, help="path to save output model weights")
    parser.add_argument("--config_path", type=str, default="configs/audioldm_m_rewas.yaml")
    args = parser.parse_args()

    assert os.path.exists(args.input_path), 'Input model does not exist.'
    assert not os.path.exists(args.output_path), 'Output filename already exists.'

    config = OmegaConf.load(args.config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{args.config_path}]')

    pretrained_weights = torch.load(args.input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()

    target_dict = {}
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
            print(f'control add: {copy_k}')
        else:
            copy_k = k
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()

    model.load_state_dict(target_dict, strict=True)
    torch.save(model.state_dict(), args.output_path)

    print(f'Model saved in {args.output_path}')

if __name__ == "__main__":
    main()