'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python3 train.py --size 1024 
                                            --batch 2 
                                            --n_sample 4 
                                            --output_dir /path/to/output/dir
                                            --warmup 0 
                                            --lambda_cycle 0.0 
                                            --lr 0.002 
                                            --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                            --iter 301 
                                            --source_class "photo" 
                                            --target_class "sketch" 
                                            --lambda_direction 1.0 
                                            --lambda_patch 0.0 
                                            --lambda_global 0.0
'''

import argparse
import os
import numpy as np

import torch

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json

from utils.file_utils import copytree, save_images

#TODO convert these to proper args
SAVE_SRC = False
SAVE_DST = True
SAVE_CYCLE_SRC = False
SAVE_CYCLE_DST = False

def train(args):

    # Set up networks, optimizers.
    net = ZSSGAN(args)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    cycle_optim = torch.optim.Adam(
        list(net.cycle_src_to_target.parameters()) + list(net.cycle_target_to_src.parameters()),
        lr = args.lr
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    fixed_z = torch.randn(args.n_sample, 512, device=device)

    for i in tqdm(range(args.iter)):

        sample_z = torch.randn(args.batch, 512, device=device)

        [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net([sample_z])

        if i < args.warmup:
            loss = cycle_loss
        else:
            loss = clip_loss + args.lambda_cycle * cycle_loss

        net.zero_grad()
        loss.backward()

        cycle_optim.step()
        if i >= args.warmup: 
            g_optim.step()


        tqdm.write(f"Clip loss: {clip_loss}, Cycle loss: {cycle_loss}, Total loss: {loss}")

        if i % 50 == 0:
            with torch.no_grad():
                [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net([fixed_z], truncation=args.sample_truncation)

                grid_rows = int(args.n_sample ** 0.5)

                if SAVE_SRC:
                    save_images(sampled_src, sample_dir, "src", grid_rows, i)

                if SAVE_DST:
                    save_images(sampled_dst, sample_dir, "dst", grid_rows, i)

                if SAVE_CYCLE_SRC:
                    save_images(cycle_src, sample_dir, "cycle_src", grid_rows, i)

                if SAVE_CYCLE_DST:
                    save_images(cycle_dst, sample_dir, "cycle_dst", grid_rows, i)

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )
            

if __name__ == "__main__":
    device = "cuda"

    torch.manual_seed(2)
    np.random.seed(2)

    parser = argparse.ArgumentParser(description="StyleGAN-NADA trainer")

    parser.add_argument(
        "--frozen_gen_ckpt", 
        type=str, 
        help="Path to a pre-trained StyleGAN2 generator for use as the initial frozen network. " \
             "If train_gen_ckpt is not provided, will also be used for the trainable generator initialization.",
        required=True)

    parser.add_argument(
        "--train_gen_ckpt", 
        type=str, 
        help="Path to a pre-trained StyleGAN2 generator for use as the initial trainable network.")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (iterations where only the cycle networks train)",
    )

    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=0.01,
        help="Strength of cycle loss",
    )

    parser.add_argument(
        "--lambda_direction",
        type=float,
        default=1.0,
        help="Strength of directional clip loss",
    )

    parser.add_argument(
        "--lambda_patch",
        type=float,
        default=0.0,
        help="Strength of patch-based clip loss",
    )

    parser.add_argument(
        "--lambda_global",
        type=float,
        default=0.0,
        help="Strength of global clip loss",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        help="How often to save a model checkpoint. No checkpoints will be saved if not set.",
    )

    parser.add_argument(
        "--source_class",
        default="dog",
        help="Textual description of the source class.",
    )

    parser.add_argument(
        "--target_class",
        default="cat",
        help="Textual description of the target class.",
    )
    
    parser.add_argument(
        "--phase",
        help="Training phase flag"
    )

    parser.add_argument(
        "--sample_truncation", 
        default=1.0,
        help="Truncation value for sampled test images."
    )

    # Original rosinality args. Most of these are not needed and should probably be removed.

    parser.add_argument(
        "--iter", type=int, default=1000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )

    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )

    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )

    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )

    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )

    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )

    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )

    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )

    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )

    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )

    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )

    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )

    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )

    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    args.train_gen_ckpt = args.train_gen_ckpt or args.frozen_gen_ckpt

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.output_dir, "code", "criteria"), )
    shutil.copy2("model/ZSSGAN.py", os.path.join(args.output_dir, "code", "ZSSGAN.py"))
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    