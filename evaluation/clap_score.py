# reference: https://github.com/Text-to-Audio/Make-An-Audio/blob/main/wav_evaluation/cal_clap_score.py

import pathlib
import sys
import os
directory = pathlib.Path(os.getcwd())
sys.path.append(str(directory))
import torch
import numpy as np
from clap.CLAPWrapper import CLAPWrapper
import argparse
from tqdm import tqdm
import pandas as pd
import json

def add_audio_path(df):
    df['audio_path'] = df.apply(lambda x:x['mel_path'].replace('.npy','.wav'),axis=1)
    return df

def build_tsv_from_wavs(root_dir, dataset):
    
    wavfiles = os.listdir(root_dir)
    # wavfiles = list(filter(lambda x:x.endswith('.wav') and x[-6:-4]!='gt',wavfiles))
    print(f'###### number of samples: {len(wavfiles)}')

    dict_list = []
    for wavfile in wavfiles:
        tmpd = {'audio_path':os.path.join(root_dir, wavfile)}
        if dataset == 'vggsound':
            caption = ' '.join(wavfile.split('_')[:-1])
        tmpd['caption'] = caption
        dict_list.append(tmpd)

    df = pd.DataFrame.from_dict(dict_list)
    tsv_path = f'{os.path.basename(root_dir)}.tsv'
    tsv_path = os.path.join('./tmp/', tsv_path)
    df.to_csv(tsv_path, sep='\t', index=False)

    return tsv_path

def cal_score_by_tsv(tsv_path, clap_model, cutoff=5):
    df = pd.read_csv(tsv_path, sep='\t')
    clap_scores = []
    if not ('audio_path' in df.columns):
        df = add_audio_path(df)
    caption_list,audio_list = [],[]
    with torch.no_grad():
        for idx,t in enumerate(tqdm(df.itertuples()), start=1): 
            caption_list.append(getattr(t,'caption'))
            audio_list.append(getattr(t,'audio_path'))
            if idx % 20 == 0:
                text_embeddings = clap_model.get_text_embeddings(caption_list)
                audio_embeddings = clap_model.get_audio_embeddings(audio_list, resample=True, cutoff=5)
                score_mat = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
                score = score_mat.diagonal()
                clap_scores.append(score.cpu().numpy())
                audio_list = []
                caption_list = []
    return np.mean(np.array(clap_scores).flatten())

def add_clap_score_to_tsv(tsv_path, clap_model):
    df = pd.read_csv(tsv_path,sep='\t')
    clap_scores_dict = {}
    with torch.no_grad():
        for idx,t in enumerate(tqdm(df.itertuples()),start=1): 
            text_embeddings = clap_model.get_text_embeddings([getattr(t,'caption')])# 经过了norm的embedding
            audio_embeddings = clap_model.get_audio_embeddings([getattr(t,'audio_path')], resample=True)
            score = clap_model.compute_similarity(audio_embeddings, text_embeddings,use_logit_scale=False)
            clap_scores_dict[idx] = score.cpu().numpy()
    df['clap_score'] = clap_scores_dict
    df.to_csv(tsv_path[:-4]+'_clap.tsv',sep='\t',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vggsound')
    parser.add_argument('--tsv_path', type=str, default='')
    parser.add_argument('--wav_dir', type=str)
    parser.add_argument('--mean', type=bool, default=True)
    parser.add_argument('--ckpt_path', default="clap")
    args = parser.parse_args()

    if args.tsv_path:
        tsv_path = args.tsv_path
    else:
        tsv_path = os.path.join('./tmp/', f'{os.path.basename(args.wav_dir)}.tsv')

    if not os.path.exists(tsv_path):
        print("result tsv not exist, build for it")
        tsv_path = build_tsv_from_wavs(args.wav_dir, args.dataset)

    clap_model = CLAPWrapper(
                    os.path.join(args.ckpt_path, 'CLAP_weights_2022.pth'),
                    os.path.join(args.ckpt_path, 'clap_config.yml'), 
                    use_cuda=True)

    clap_score = cal_score_by_tsv(tsv_path, clap_model, cutoff=5)
    out = args.wav_dir if args.wav_dir else args.tsv_path

    print(f"Clap score for {out} is:{clap_score}")