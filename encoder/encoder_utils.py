# reference: https://github.com/v-iashin/Synchformer

import torch
import torch.distributed as dist
import torch.nn as nn
import importlib
from tqdm import tqdm 
from pathlib import Path
import torchvision
from omegaconf import OmegaConf
import subprocess

PARENT_LINK = 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
FNAME2LINK = {
    # S3: Synchability: AudioSet (run 2)
    '24-01-22T20-34-52.pt': f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/24-01-22T20-34-52.pt',
    'cfg-24-01-22T20-34-52.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-22T20-34-52/cfg-24-01-22T20-34-52.yaml',
    # S2: Synchformer: AudioSet (run 2)
    '24-01-04T16-39-21.pt': f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt',
    'cfg-24-01-04T16-39-21.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml',
    # S2: Synchformer: AudioSet (run 1)
    '23-08-28T11-23-23.pt': f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/23-08-28T11-23-23.pt',
    'cfg-23-08-28T11-23-23.yaml': f'{PARENT_LINK}/sync/sync_models/23-08-28T11-23-23/cfg-23-08-28T11-23-23.yaml',
    # S2: Synchformer: LRS3 (run 2)
    '23-12-23T18-33-57.pt': f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.pt',
    'cfg-23-12-23T18-33-57.yaml': f'{PARENT_LINK}/sync/sync_models/23-12-23T18-33-57/cfg-23-12-23T18-33-57.yaml',
    # S2: Synchformer: VGS (run 2)
    '24-01-02T10-00-53.pt': f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/24-01-02T10-00-53.pt',
    'cfg-24-01-02T10-00-53.yaml': f'{PARENT_LINK}/sync/sync_models/24-01-02T10-00-53/cfg-24-01-02T10-00-53.yaml',
    # SparseSync: ft VGGSound-Full
    '22-09-21T21-00-52.pt': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/22-09-21T21-00-52.pt',
    'cfg-22-09-21T21-00-52.yaml': f'{PARENT_LINK}/sync/sync_models/22-09-21T21-00-52/cfg-22-09-21T21-00-52.yaml',
    # SparseSync: ft VGGSound-Sparse
    '22-07-28T15-49-45.pt': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/22-07-28T15-49-45.pt',
    'cfg-22-07-28T15-49-45.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-28T15-49-45/cfg-22-07-28T15-49-45.yaml',
    # SparseSync: only pt on LRS3
    '22-07-13T22-25-49.pt': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/22-07-13T22-25-49.pt',
    'cfg-22-07-13T22-25-49.yaml': f'{PARENT_LINK}/sync/sync_models/22-07-13T22-25-49/cfg-22-07-13T22-25-49.yaml',
    # SparseSync: feature extractors
    'ResNetAudio-22-08-04T09-51-04.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-04T09-51-04.pt',  # 2s
    'ResNetAudio-22-08-03T23-14-49.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-49.pt',  # 3s
    'ResNetAudio-22-08-03T23-14-28.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-08-03T23-14-28.pt',  # 4s
    'ResNetAudio-22-06-24T08-10-33.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T08-10-33.pt',  # 5s
    'ResNetAudio-22-06-24T17-31-07.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T17-31-07.pt',  # 6s
    'ResNetAudio-22-06-24T23-57-11.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-24T23-57-11.pt',  # 7s
    'ResNetAudio-22-06-25T04-35-42.pt': f'{PARENT_LINK}/sync/ResNetAudio-22-06-25T04-35-42.pt',  # 8s
}



def check_if_file_exists_else_download(path, fname2link=FNAME2LINK, chunk_size=1024):
    '''Checks if file exists, if not downloads it from the link to the path'''
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        with requests.get(fname2link[path.name], stream=True) as r:
            total_size = int(r.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                with open(path, 'wb') as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



def get_model(cfg, device):
    model = instantiate_from_config(cfg.model)

    # TODO: maybe in the module
    if cfg.model.params.vfeat_extractor.is_trainable is False:
        for params in model.vfeat_extractor.parameters():
            params.requires_grad = False
    if cfg.model.params.afeat_extractor.is_trainable is False:
        for params in model.afeat_extractor.parameters():
            params.requires_grad = False

    model = model.to(device)
    model_without_ddp = model
    # print(dist.is_initialized())
    if dist.is_initialized():
        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[cfg.training.local_rank])
        # any mistaken calls on `model_without_ddp` (=None) will likely raise an error
        model_without_ddp = model.module
    # input()
    return model, model_without_ddp



def get_pretrained(exp_name, local_rank=None):
        
    cfg_path = f'./configs/cfg-{exp_name}.yaml'
    ckpt_path = f'./ckpts/{exp_name}.pt'

    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    cfg = OmegaConf.load(cfg_path)

    # patch config
    cfg = patch_config(cfg)
    cfg.model.params.vfeat_extractor.is_trainable = False
    try:
        _, synch_model = get_model(cfg, f'cuda:{local_rank}')
    except:
        _, synch_model = get_model(cfg, f'cuda:0')
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    synch_model.load_state_dict(ckpt['model'])

    vfeat_extractor = synch_model.vfeat_extractor

    vfeat_extractor.eval()

    return vfeat_extractor

def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg

def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path
    
def get_transforms(cfg, which_transforms=['train', 'test'], half=True):
    transforms = {}
    for mode in which_transforms:
        ts_cfg = cfg.get(f'transform_sequence_{mode}', None)
        if mode == 'test':
            if half:
                remove_audio = [
                    {'target': 'encoder.transforms.EqualifyFromRight'}, 
                    {'target': 'encoder.transforms.AudioMelSpectrogram', 'params': {'sample_rate': 16000, 'win_length': 400, 'hop_length': 160, 'n_fft': 1024, 'n_mels': 128}}, 
                    {'target': 'encoder.transforms.AudioLog'}, 
                    {'target': 'encoder.transforms.PadOrTruncate', 'params': {'max_spec_t': 66}}, 
                    {'target': 'encoder.transforms.AudioNormalizeAST', 'params': {'mean': -4.2677393, 'std': 4.5689974}}
                    ]
                ts_cfg = [x for x in ts_cfg if x not in remove_audio]
            else:
                only_video = [
                    {'target': 'encoder.transforms.RGBSpatialCrop', 'params': {'input_size': 112, 'is_random': False}}, 
                    {'target': 'encoder.transforms.RGBToHalfToZeroOne'}, 
                    {'target': 'encoder.transforms.RGBNormalize', 'params': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}} 
                    ]
                ts_cfg = [x for x in ts_cfg if x in only_video]

        ts = [lambda x: x] if ts_cfg is None else [instantiate_from_config(c) for c in ts_cfg]

        transforms[mode] = torchvision.transforms.Compose(ts)

    return transforms


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    '''If the `model` object is wrapped in `torch.nn.parallel.DistributedDataParallel` we have
    to use `model.modules` to get access to methods of the model. This wrapper allows
    to avoid using `if ddp: model.module.* else: model.*`. Used during `evaluate_on_sync_w_shifts`.'''
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
