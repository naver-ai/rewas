import os
import json

import torch
import numpy as np

import audioldm.hifigan as hifigan

import importlib

import torch
import numpy as np
from collections import abc

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


HIFIGAN_16K_64 = {
    "resblock": "1",
    "num_gpus": 6,
    "batch_size": 16,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,
    "upsample_rates": [5, 4, 2, 2, 2],
    "upsample_kernel_sizes": [16, 16, 8, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "segment_size": 8192,
    "num_mels": 64,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 160,
    "win_size": 1024,
    "sampling_rate": 16000,
    "fmin": 0,
    "fmax": 8000,
    "fmax_for_loss": None,
    "num_workers": 4,
    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1,
    },
}


def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_text(latent_diffusion, control = True):
    if not control:
        latent_diffusion.cond_stage_key = "text"
        latent_diffusion.cond_stage_model.embed_mode="text"
    else: 
        latent_diffusion.cond_stage_key = "text"
        model_idx = latent_diffusion.cond_stage_model_metadata['film_clap_cond1']["model_idx"]
        latent_diffusion.cond_stage_models[model_idx].embed_mode = "text"
    return latent_diffusion
 
 
def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

# def get_obj_from_str(string, reload=False):
#     module, cls = string.rsplit(".", 1)
#     if reload:
#         module_imp = importlib.import_module(module)
#         importlib.reload(module_imp)
#     return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
    func: callable,
    data,
    n_proc,
    target_data_type="ndarray",
    cpu_intensive=True,
    use_worker_id=False,
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i : i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == "ndarray":
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == "list":
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res


def get_available_checkpoint_keys(model, ckpt):
    print("==> Attemp to reload from %s" % ckpt)
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if "_orig_mod." in key:
            new_state_dict["generator"][key.replace("_orig_mod.", "")] = state_dict[
                "generator"
            ][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict


def get_vocoder(config, device, ckpt_path):
    config = hifigan.AttrDict(HIFIGAN_16K_64)
    vocoder = hifigan.Generator(config)

    assert os.path.exists(ckpt_path), print(f'Check the file path: {ckpt_path}')

    ckpt = torch.load(ckpt_path)
    ckpt = torch_version_orig_mod_remove(ckpt)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    return wavs
