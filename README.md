## Read, Watch and Scream! Sound Generation from Text and Video

[![arXiv](https://img.shields.io/badge/arXiv%20papr-2407.05551-b31b1b.svg)](https://arxiv.org/abs/2407.05551)
[![Samples](https://img.shields.io/badge/Demo-Link-blue.svg)](https://naver-ai.github.io/rewas/)


[Yujin Jeong](https://eugene6923.github.io/)&nbsp; [Yunji Kim](https://github.com/YunjiKim)&nbsp; [Sanghyuk Chun](https://sanghyukchun.github.io/home/)&nbsp; [Jiyoung Lee](https://lee-jiyoung.github.io/)

NAVER AI Lab

--- 

### We release the official code!

---

### Abstract


Multimodal generative models have shown impressive advances with the help of powerful diffusion models.
Despite the progress, generating sound solely from text poses challenges in ensuring comprehensive scene depiction and temporal alignment.
Meanwhile, video-to-sound generation limits the flexibility to prioritize sound synthesis for specific objects within the scene.
To tackle these challenges, we propose a novel video-and-text-to-sound generation method, called **ReWaS**, where video serves as a conditional control for a text-to-audio generation model.
Our method estimates the structural information of audio (namely, energy) from the video while receiving key content cues from a user prompt.
We employ a well-performing text-to-sound model to consolidate the video control, which is much more efficient for training multimodal diffusion models with massive triplet-paired (audio-video-text) data.
In addition, by separating the generative components of audio, it becomes a more flexible system that allows users to freely adjust the energy, surrounding environment, and primary sound source according to their preferences.
Experimental results demonstrate that our method shows superiority in terms of quality, controllability, and training efficiency.




## ReWaS

### Prepare Python running environment

```shell 
git clone https://github.com/naver-ai/rewas.git
# Install running environment
sudo apt-get update
sudo apt-get install -y python3-tk
sudo apt-get install -y ffmpeg
pip install -r requirements.txt
```

If the code raises the following error, 'No module named 'pytorch_lightning.utilities.rank_zero', please upgrade pytorch-lightning.

### Download checkpoints

1. Download checkpoints from [link](https://huggingface.co/lee-j/ReWaS/tree/main) that contains parameteres of ReWaS(AudioLDM-M) and phi.

2. Download the checkpoints of pretrained Synchformer, VAE, CLAP, 16kHz HiFiGAN, and 48kHz HiFiGAN from [Synchformer](https://github.com/v-iashin/Synchformer?tab=readme-ov-file#audio-visual-synchronization-models) and [AudioLDM-training](https://github.com/haoheliu/AudioLDM-training-finetuning?tab=readme-ov-file#download-checkpoints-and-dataset).


```shell
ckpts/
  vae_mel_16k_64bins.ckpt
  hifigan_16k_64bins.ckpt
  clap_music_speech_audioset_epoch_15_esc_89.98.pt
  24-01-04T16-39-21.pt
  phi_vggsound.ckpt
  audioldm_m_rewas_vggsound.ckpt
```

### Test ReWaS
Please insert the video path and text prompt that you want to generate audio into 'test_samples.json'.

Use the following syntax:

```shell
python test.py \
  -ckpt ckpts/rewas.ckpt \
  --config configs/audioldm_m_rewas.yaml \
  --control_type energy_video \
  --save_path outputs \
  --testlist 'test_samples.json'
```

### Evaluate model

We recommend the following evaluation metrics.

1. **Energy MAE**: ./eval_MAE.py
2. [**Melception Audio Quality**](https://github.com/v-iashin/SpecVQGAN/blob/main/evaluate.py)
3. [**CLAP score**](https://github.com/Text-to-Audio/Make-An-Audio/tree/main/wav_evaluation) 
- Download CLAP weights from [Hugging Face](https://huggingface.co/microsoft/msclap/blob/main/CLAP_weights_2022.pth) into `evaluation/clap/CLAP_weights_2022.pth`
  ```shell 
  cd evaluation;
  python clap_score.py
  ```
- requirements: transformer>=4.28.1

4. [**Onset Accuracy**](https://github.com/XYPB/CondFoleyGen/blob/main/predict_onset.py)
5. [**AV-align**](https://github.com/guyyariv/TempoTokens/blob/master/av_align.py)
    ```shell
    cd evaluation;
    python av_align_score.py --input_video_dir='/path/to/vggsound_video' --input_wav_dir='results/' --cache_path='./video_cache.json'
    ```

### Customizing
If you want to build a new ReWaS or apply in other text-to-audio model, you can use `tool_add_adapter.py`


## BibTex

```
@inproceedings{jeong2024read,
  author    = {Jeong, Yujin and Kim, Yunji and Chun, Sanghyuk and Lee, Jiyoung},
  title     = {Read, Watch and Scream! Sound Generation from Text and Video},
  journal   = {arXiv preprint arXiv:2407.05551},
  year      = {2024},
}
```

## License
```
ReWaS
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
```

## Reference
We greatly appreciate the open-soucing of the following code bases. Open source code base is the real-world infinite stone 💎!
- https://github.com/haoheliu/AudioLDM-training-finetuning
- https://github.com/lllyasviel/ControlNet
- https://github.com/v-iashin/Synchformer

