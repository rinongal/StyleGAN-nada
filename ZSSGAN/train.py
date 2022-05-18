'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024 
                                           --batch 2 
                                           --n_sample 4 
                                           --output_dir /path/to/output/dir 
                                           --lr 0.002 
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                           --iter 301 
                                           --source_class "photo" 
                                           --target_class "sketch" 
                                           --lambda_direction 1.0 
                                           --lambda_patch 0.0 
                                           --lambda_global 0.0 
                                           --lambda_texture 0.0 
                                           --lambda_manifold 0.0 
                                           --phase None 
                                           --auto_layer_k 0 
                                           --auto_layer_iters 0 
                                           --auto_layer_batch 8 
                                           --output_interval 50 
                                           --clip_models "ViT-B/32" "ViT-B/16" 
                                           --clip_model_weights 1.0 1.0 
                                           --mixing 0.0
                                           --save_interval 50
'''

import argparse
import os
import numpy as np

import torch

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json
import pickle
import copy

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True

def train(args):

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    z_dim = 64 if args.sgxl else 512

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(2)
    np.random.seed(2)

    # Training loop
    fixed_z = torch.randn(args.n_sample, z_dim, device=device)

    for i in tqdm(range(args.iter)):

        net.train()

        sample_z = mixing_noise(args.batch, z_dim, args.mixing, device)

        [sampled_src, sampled_dst], loss = net(sample_z)

        net.zero_grad()
        loss.backward()

        g_optim.step()

        tqdm.write(f"Clip loss: {loss}")

        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src, sampled_dst], loss = net([fixed_z], truncation=args.sample_truncation)

                if args.crop_for_cars:
                    sampled_dst = sampled_dst[:, :, 64:448, :]

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):

            if args.sg3 or args.sgxl:

                snapshot_data = {'G_ema': copy.deepcopy(net.generator_trainable.generator).eval().requires_grad_(False).cpu()}
                snapshot_pkl = f'{ckpt_dir}/{str(i).zfill(6)}.pkl'

                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

            else:
                torch.save(
                    {
                        "g_ema": net.generator_trainable.generator.state_dict(),
                        "g_optim": g_optim.state_dict(),
                    },
                    f"{ckpt_dir}/{str(i).zfill(6)}.pt",
                )

    for i in range(args.num_grid_outputs):
        net.eval()

        with torch.no_grad():
            sample_z = mixing_noise(16, z_dim, 0, device)
            [sampled_src, sampled_dst], _ = net(sample_z, truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")
            

if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    