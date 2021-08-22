'''
Tool for generating editing videos across different domains.

Given a set of latent codes and pre-trained models, it will interpolate between the different codes in each of the target domains
and combine the resulting images into a video.

Example run command:

python generate_videos.py --ckpt /model_dir/pixar.pt \
                                 /model_dir/ukiyoe.pt \
                                 /model_dir/edvard_munch.pt \
                                 /model_dir/botero.pt \
                          --out_dir /output/video/ \
                          --source_latent /latents/latent000.npy \
                          --target_latents /latents/

'''

import os
import argparse

import torch
from torchvision import utils

from model.sg2_model import Generator
from tqdm import tqdm
import numpy as np

import subprocess
import shutil

def generate_interpolations(args, g_ema, output_dir):
    source_latent = np.load(args.source_latent, allow_pickle=True)

    alphas = np.linspace(0, 1, num=40)
    
    # interpolate latent codes with all targets

    print("Interpolating latent codes...")
    latents = []
    for target_latent_path in args.target_latents:
        
        if target_latent_path == args.source_latent:
            continue

        target_latent = np.load(target_latent_path, allow_pickle=True)

        latents_forward  = [a * target_latent + (1-a) * source_latent for a in alphas] # interpolate from source to target
        latents_backward = latents_forward[::-1]                                       # interpolate from target to source
        latents.extend(latents_forward + [target_latent] * 12 + latents_backward)      # forward + short delay at target + return

    # generte images for all interpolations

    print("Generating frames for video...")
    for idx, latent in tqdm(enumerate(latents), total=len(latents)):

        w = torch.from_numpy(latent).float().cuda()

        with torch.no_grad():
            img, _ = g_ema([w], input_is_latent=True, truncation=1, randomize_noise=False)
            utils.save_image(img, f"{output_dir}/{str(idx).zfill(3)}.jpg", nrow=1, normalize=True, scale_each=True, range=(-1, 1))

def video_from_interpolations(output_dir):
    # combine frames to a video
    command = ["ffmpeg", 
               "-r", "25", 
               "-i", f"{output_dir}/%03d.jpg", 
               "-c:v", "libx264", 
               "-vf", "fps=25", 
               "-pix_fmt", "yuv420p", 
               f"{output_dir}/out.mp4"]
    
    subprocess.call(command)

def merge_videos(output_dir, num_subdirs):

    output_file = os.path.join(output_dir, "combined.mp4")

    if num_subdirs == 1: # if we only have one video, just copy it over
        shutil.copy2(os.path.join(output_dir, str(0), "out.mp4"), output_file)
    else:                # otherwise merge using ffmpeg
        command = ["ffmpeg"]
        for dir in range(num_subdirs):
            command.extend(['-i', os.path.join(output_dir, str(dir), "out.mp4")])
        
        sqrt_subdirs = int(num_subdirs ** .5)

        if (sqrt_subdirs ** 2) != num_subdirs:
            raise ValueError("Number of checkpoints cannot be arranged in a square grid")
        
        command.append("-filter_complex")

        filter_string = ""
        vstack_string = ""
        for row in range(sqrt_subdirs):
            row_str = ""
            for col in range(sqrt_subdirs):
                row_str += f"[{row * sqrt_subdirs + col}:v]"

            letter = chr(ord('A')+row)
            row_str += f"hstack=inputs={sqrt_subdirs}[{letter}];"
            vstack_string += f"[{letter}]"

            filter_string += row_str
        
        vstack_string += f"vstack=inputs={sqrt_subdirs}[out]"
        filter_string += vstack_string

        command.extend([filter_string, "-map", "[out]", output_file])

        subprocess.call(command)

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, nargs="+", required=True, help="Path to one or more pre-trained generator checkpoints.")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--out_dir', type=str, required=True, help="Directory where output files will be placed")
    parser.add_argument('--source_latent', type=str, required=True, help="Path to an .npy file containing an initial latent code")
    parser.add_argument('--target_latents', nargs="+", type=str, help="A list of paths to .npy files containing target latent codes to interpolate towards, or a directory containing such .npy files.")
    parser.add_argument('--force', '-f', action='store_true', help="Force run with non-empty directory. Image files not overwritten by the proccess may still be included in the final video")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.force and os.listdir(args.out_dir):
        print("Output directory is not empty. Either delete the directory content or re-run with -f.")
        exit(0)

    if len(args.target_latents) == 1 and os.path.isdir(args.target_latents[0]):
        args.target_latents = [os.path.join(args.target_latents[0], file_name) for file_name in os.listdir(args.target_latents[0]) if file_name.endswith(".npy")]
        args.target_latents = sorted(args.target_latents)

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    for idx, ckpt_path in enumerate(args.ckpt):
        print(f"Generating video using checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

        g_ema.load_state_dict(checkpoint['g_ema'])

        output_dir = os.path.join(args.out_dir, str(idx))
        os.makedirs(output_dir)

        generate_interpolations(args, g_ema, output_dir)
        video_from_interpolations(output_dir)

    merge_videos(args.out_dir, len(args.ckpt))

    

