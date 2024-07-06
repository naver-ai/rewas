## Read, Watch and Scream! Sound Generation from Text and Video

### The code will be coming soon!

[Yujin Jeong]()&nbsp; [Yunji Kim](https://github.com/YunjiKim)&nbsp; [Sanghyuk Chun](https://sanghyukchun.github.io/home/)&nbsp; [Jiyoung Lee](https://lee-jiyoung.github.io/)

NAVER AI Lab

### Abstract


Multimodal generative models have shown impressive advances with the help of powerful diffusion models.
Despite the progress, generating sound solely from text poses challenges in ensuring comprehensive scene depiction and temporal alignment.
Meanwhile, video-to-sound generation limits the flexibility to prioritize sound synthesis for specific objects within the scene.
To tackle these challenges, we propose a novel video-and-text-to-sound generation method, called **ReWaS**, where video serves as a conditional control for a text-to-audio generation model.
Our method estimates the structural information of audio (namely, energy) from the video while receiving key content cues from a user prompt.
We employ a well-performing text-to-sound model to consolidate the video control, which is much more efficient for training multimodal diffusion models with massive triplet-paired (audio-video-text) data.
In addition, by separating the generative components of audio, it becomes a more flexible system that allows users to freely adjust the energy, surrounding environment, and primary sound source according to their preferences.
Experimental results demonstrate that our method shows superiority in terms of quality, controllability, and training efficiency.



## BibTex

```
@inproceedings{jeong2024read,
  author    = {Jeong, Yujin and Kim, Yunji and Chun, Sanghyuk and Lee, Jiyoung},
  title     = {Read, Watch and Scream! Sound Generation from Text and Video},
  year      = {2024},
}
```