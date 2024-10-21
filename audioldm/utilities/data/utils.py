import os
from pathlib import Path
import subprocess
import random
import numpy as np
import soundfile as sf
import torch
import torchvision
from moviepy.editor import VideoFileClip, AudioFileClip

from encoder.encoder_utils import which_ffmpeg

def get_video_and_audio(path, get_meta=False, duration=5, start_sec=0, end_sec=None, random_start = False):
    rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, 'sec', output_format='TCHW')
    assert meta['video_fps'], f'No video fps for {path}'
    
    vlen = int(duration * meta['video_fps'])
    meta_out = {'video': {'fps': [meta['video_fps']]}}
    

    if random_start:
        stx = random.randint(0, int(rgb.size(0)/meta['video_fps']) - duration)
    else:
        stx = 0
        
    rgb = rgb[int(stx*meta['video_fps']):int(stx*meta['video_fps']+vlen), :, :, :]

    if rgb.shape[0] < vlen:

        rgb = torch.cat([rgb, rgb, rgb], dim=0)
        rgb = rgb[int(stx*meta['video_fps']):int(stx*meta['video_fps']+vlen), :, :, :]


    if meta.get('audio_fps'):
        alen = int(duration * meta['audio_fps'])
        audio = audio.mean(dim=0)
        audio = audio[stx*meta['audio_fps']:(stx*meta['audio_fps']+alen)]
        meta_out['audio'] =  {'framerate': [meta['audio_fps']]}
    else:
        meta_out['audio'] =  {'framerate': [16000]}

    return rgb, audio, meta_out


def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    paths = []
    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s_%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            ),
        )
        paths.append(path)
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)
        
    return paths

def save_video(audio_path, video_path):

    video_clip = VideoFileClip(video_path)
    video_clip = video_clip.subclip(0, 5) # generated audio duration is 5 seconds.

    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    
    # Output file path for the final video with audio
    out_video_path = audio_path.replace('.wav', '.mp4')

    # Write the video clip with the audio to a new file
    video_clip.write_videofile(out_video_path, audio_codec='aac')

    # Close the clips
    video_clip.close()
    audio_clip.close()

    return
        
def re_encode_video(new_path, path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    
    os.makedirs(new_path, exist_ok=True)

    new_path += f"/{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    new_path = str(new_path)
    cmd = f"{which_ffmpeg()}"
    # no info/error printing
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -y -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f" {new_path}"
    subprocess.call(cmd.split())
    return new_path
