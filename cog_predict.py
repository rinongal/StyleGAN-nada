import time

t_start = time.time()

import os
import sys
import tempfile
import shutil
from argparse import Namespace
from pathlib import Path

import cog
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import utils
from PIL import Image


sys.path.insert(0, "/content")
sys.path.insert(0, "encoder4editing")

sys.path.insert(0, "ZSSGAN")

from model.sg2_model import Generator
from generate_videos import generate_frames, video_from_interpolations, vid_to_gif

from encoder4editing.models.psp import pSp
from encoder4editing.utils.alignment import align_face
from encoder4editing.utils.common import tensor2im

model_list = ['base'] + [Path(model_ckpt).stem for model_ckpt in os.listdir("models") if not 'base' in model_ckpt]

class Predictor(cog.Predictor):
    def setup(self):

        print("starting setup")
        t_start_setup = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        latent_size = 512
        n_mlp = 8
        channel_mult = 2
        model_size = 1024

        self.generators = {}

        for model in model_list:
            g_ema = Generator(
                model_size, latent_size, n_mlp, channel_multiplier=channel_mult
            ).to(self.device)

            checkpoint = torch.load(f"models/{model}.pt")

            g_ema.load_state_dict(checkpoint['g_ema'])

            self.generators[model] = g_ema

        self.experiment_args = {"model_path": "e4e_ffhq_encode.pt"}
        self.experiment_args["transform"] = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.resize_dims = (256, 256)

        model_path = self.experiment_args["model_path"]

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        # pprint.pprint(opts)  # Display full options used
        # update the training options
        opts["checkpoint_path"] = model_path
        opts = Namespace(**opts)

        self.e4e_net = pSp(opts)
        self.e4e_net.eval()
        self.e4e_net.cuda()

        self.shape_predictor = dlib.shape_predictor(
            "/content/shape_predictor_68_face_landmarks.dat"
        )
        t_end = time.time()

        self.time_gap = t_end - t_start
        self.time_gap_setup = t_end - t_start_setup
        print("setup complete")

    @cog.input("input", type=Path, help="Input image")
    @cog.input("output_style",
               type=str,
               help=f"Which output style do you want to use? Select 'all' to generate a collage.",
               options=model_list + ['all'] + ['list - enter below'],
               default='all')
    @cog.input("style_list", type=str, default='joker,anime,modigliani', help="Comma seperated list of models to use. Only accepts models from the output_style list. Will only be used if the chosen output_style is list")
    @cog.input("generate_video", type=bool, default=False, help="Generate a video instead of an output image. If more than one style is used, will interpolate between styles.")
    @cog.input("with_editing", type=bool, default=True, help="Apply latent space editing to the generated video")
    @cog.input("video_format", type=str, help="Choose gif to display in browser, mp4 for a higher-quality downloadable video", options=['gif', 'mp4'], default='mp4')
    def predict(
        self,
        input,
        output_style,
        style_list,
        generate_video,
        with_editing,
        video_format
    ):  

        if output_style == 'all':
            styles = model_list
        elif output_style == 'list - enter below':
            styles = style_list.split(",")
            for style in styles:
                if style not in model_list:
                    raise ValueError(f"Encountered style '{style}' in the style_list which is not an available option.")
        else:
            styles = [output_style]

        # @title Align image
        input_image = self.run_alignment(str(input))
        
        input_image = input_image.resize(self.resize_dims)

        img_transforms = self.experiment_args["transform"]
        transformed_image = img_transforms(input_image)

        with torch.no_grad():
            images, latents = self.run_on_batch(transformed_image.unsqueeze(0))
            result_image, latent = images[0], latents[0]

        inverted_latent = latent.unsqueeze(0).unsqueeze(1)
        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "out.jpg"
        
        generators = [self.generators[style] for style in styles]

        if not generate_video:
            with torch.no_grad():
                img_list = []
                for g_ema in generators:
                    img, _ = g_ema(inverted_latent, input_is_latent=True, truncation=1, randomize_noise=False)
                    img_list.append(img)
                
                out_img = torch.cat(img_list, axis=0)
                utils.save_image(out_img, out_path, nrow=int(np.sqrt(out_img.size(0))), normalize=True, scale_each=True, range=(-1, 1))

            return Path(out_path)
        
        return self.generate_vid(generators, inverted_latent, out_dir, video_format, with_editing)

    def generate_vid(self, generators, latent, out_dir, video_format, with_editing):      
        np_latent = latent.squeeze(0).cpu().detach().numpy()
        args = {
                'fps': 24,
                'target_latents': None,
                'edit_directions': None,
                'unedited_frames': 0 if with_editing else 40 * (len(generators) - 1)
                }

        args = Namespace(**args)
        with tempfile.TemporaryDirectory() as dirpath:

            generate_frames(args, np_latent, generators, dirpath)
            video_from_interpolations(args.fps, dirpath)
            
            gen_path = Path(dirpath) / "out.mp4"
            out_path = out_dir / f"out.{video_format}"

            if video_format == 'gif':
                vid_to_gif(gen_path, out_dir, scale=256, fps=args.fps)
            else:
                shutil.copy2(gen_path, out_path) 

        return out_path

    def run_alignment(self, image_path):
        aligned_image = align_face(filepath=image_path, predictor=self.shape_predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image

    def run_on_batch(self, inputs):
        images, latents = self.e4e_net(
            inputs.to("cuda").float(), randomize_noise=False, return_latents=True
        )
        return images, latents