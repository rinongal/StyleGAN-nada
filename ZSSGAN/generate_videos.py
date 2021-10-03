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
from pathlib import Path

import numpy as np

import subprocess
import shutil
import copy

VALID_EDITS = ["pose", "age", "smile", "gender", "hair_length", "beard"]

SUGGESTED_DISTANCES = {
                       "pose": (3.0, -3.0),
                       "smile": (2.0, -2.0),
                       "age": (4.0, -4.0),
                       "gender": (3.0, -3.0),
                       "hair_length": (None, -4.0),
                       "beard": (2.0, None)
                      }
                    
def project_code(latent_code, boundary, distance=3.0):

    if len(boundary) == 2:
        boundary = boundary.reshape(1, 1, -1)

    return latent_code + distance * boundary

def generate_frames(args, source_latent, g_ema_list, output_dir):

    alphas = np.linspace(0, 1, num=20)
    
    interpolate_func = interpolate_with_boundaries # default
    if args.target_latents:                        # if provided with targets
        interpolate_func = interpolate_with_target_latents
    if args.unedited_frames:                       # if only interpolating through generators
        interpolate_func = duplicate_latent

    latents = interpolate_func(args, source_latent, alphas)

    segments = len(g_ema_list) - 1
    if segments:
        segment_length = len(latents) / segments

        g_ema = copy.deepcopy(g_ema_list[0])

        src_pars = dict(g_ema.named_parameters())
        mix_pars = [dict(model.named_parameters()) for model in g_ema_list]
    else:
        g_ema = g_ema_list[0]

    print("Generating frames for video...")
    for idx, latent in tqdm(enumerate(latents), total=len(latents)):

        if segments:
            mix_alpha = (idx % segment_length) * 1.0 / segment_length
            segment_id = int(idx // segment_length)

            for k in src_pars.keys():
                src_pars[k].data.copy_(mix_pars[segment_id][k] * (1 - mix_alpha) + mix_pars[segment_id + 1][k] * mix_alpha)

        if idx == 0 or segments or latent is not latents[idx - 1]:
            w = torch.from_numpy(latent).float().cuda()

            with torch.no_grad():
                img, _ = g_ema([w], input_is_latent=True, truncation=1, randomize_noise=False)

        utils.save_image(img, f"{output_dir}/{str(idx).zfill(3)}.jpg", nrow=1, normalize=True, scale_each=True, range=(-1, 1))

def interpolate_forward_backward(source_latent, target_latent, alphas):
    latents_forward  = [a * target_latent + (1-a) * source_latent for a in alphas] # interpolate from source to target
    latents_backward = latents_forward[::-1]                                       # interpolate from target to source
    return latents_forward + [target_latent] * 20 + latents_backward               # forward + short delay at target + return

def duplicate_latent(args, source_latent, alphas):
    return [source_latent for _ in range(args.unedited_frames)]

def interpolate_with_boundaries(args, source_latent, alphas):
    edit_directions = args.edit_directions or ['pose', 'smile', 'gender', 'age', 'hair_length']
    
    # interpolate latent codes with all targets

    print("Interpolating latent codes...")

    boundary_dir = Path(os.path.abspath(__file__)).parents[1].joinpath("editing", "interfacegan_boundaries")

    boundaries_and_distances = []
    for direction_type in edit_directions:
        distances = SUGGESTED_DISTANCES[direction_type]
        boundary = torch.load(os.path.join(boundary_dir, f'{direction_type}.pt')).cpu().detach().numpy()

        for distance in distances:
            if distance:
                boundaries_and_distances.append((boundary, distance))

    latents = []
    for boundary, distance in boundaries_and_distances:
        
        target_latent = project_code(source_latent, boundary, distance)
        latents.extend(interpolate_forward_backward(source_latent, target_latent, alphas)) 

    return latents

def interpolate_with_target_latents(args, source_latent, alphas):    
    # interpolate latent codes with all targets

    print("Interpolating latent codes...")
    
    latents = []
    for target_latent_path in args.target_latents:
        
        if target_latent_path == args.source_latent:
            continue

        target_latent = np.load(target_latent_path, allow_pickle=True)

        latents.extend(interpolate_forward_backward(source_latent, target_latent, alphas))

    return latents

def video_from_interpolations(fps, output_dir):

    # combine frames to a video
    command = ["ffmpeg", 
               "-r", f"{fps}", 
               "-i", f"{output_dir}/%03d.jpg", 
               "-c:v", "libx264", 
               "-vf", f"fps={fps}", 
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

def vid_to_gif(vid_path, output_dir, scale=256, fps=35):

    command = ["ffmpeg", 
               "-i", f"{vid_path}", 
               "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1]fifo[s2];[s2][p]paletteuse",  
               "-loop", "0",
               f"{output_dir}/out.gif"]
    
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
    parser.add_argument('--fps', default=35, type=int, help='Frames per second in the generated videos.')
    parser.add_argument('--edit_directions', nargs="+", type=str, help=f"A list of edit directions to use in video generation (if not using a target latent directory). Available directions are: {VALID_EDITS}")
    parser.add_argument('--unedited_frames', type=int, default=0, help="Used to generate videos with no latent editing. If set to a positive number and target_latents is not provided, will simply duplicate the initial frame <unedited_frames> times.")
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.force and os.listdir(args.out_dir):
        print("Output directory is not empty. Either delete the directory content or re-run with -f.")
        exit(0)

    if args.target_latents and len(args.target_latents) == 1 and os.path.isdir(args.target_latents[0]):
        args.target_latents = [os.path.join(args.target_latents[0], file_name) for file_name in os.listdir(args.target_latents[0]) if file_name.endswith(".npy")]
        args.target_latents = sorted(args.target_latents)

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    source_latent = np.load(args.source_latent, allow_pickle=True)

    for idx, ckpt_path in enumerate(args.ckpt):
        print(f"Generating video using checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)

        g_ema.load_state_dict(checkpoint['g_ema'])

        output_dir = os.path.join(args.out_dir, str(idx))
        os.makedirs(output_dir)

        generate_frames(args, source_latent, [g_ema], output_dir)
        video_from_interpolations(args.fps, output_dir)

    merge_videos(args.out_dir, len(args.ckpt))

    

