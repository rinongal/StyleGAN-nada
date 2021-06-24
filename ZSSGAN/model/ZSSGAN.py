import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../pytorch-CycleGAN-and-pix2pix'))


import torch
import torchvision.transforms as transforms

import numpy as np

from functools import partial

from pytorch_CycleGAN_and_pix2pix.models import networks as CycleGAN
from ZSSGAN.model.sg2_model import Generator, Discriminator
from ZSSGAN.criteria.clip_loss import CLIPLoss       

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Generator, self).__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers()[4][0:2])  
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers() 
        else: 
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    #TODO Maybe convert to kwargs
    def forward(self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True):
        return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent)

class SG2Discriminator(torch.nn.Module):
    def __init__(self, checkpoint_path, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Discriminator, self).__init__()

        self.discriminator = Discriminator(
            img_size, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.discriminator.load_state_dict(checkpoint["d"], strict=True)

    def get_all_layers(self):
        return list(self.discriminator.children())

    def get_training_layers(self):
        return self.get_all_layers() 

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def forward(self, images):
        return self.discriminator(images)

class ZSSGAN(torch.nn.Module):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        device = 'cuda:0'

        # Set up frozen (source) generator
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size).to(device)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # discriminator is currently unused. TODO: re-enable as required or remove.

        # self.discriminator_frozen = SG2Discriminator(args.frozen_gen_ckpt, img_size=512).to(device)
        # self.discriminator_frozen.freeze_layers()
        # self.discriminator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size).to(device)
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()

        # Set up cycle networks
        self.cycle_target_to_src = CycleGAN.define_G(3, 3, 64, "unet_256",  "instance", True, "normal", 0.02, [0]).to(device)
        self.cycle_src_to_target = CycleGAN.define_G(3, 3, 64, "unet_256",  "instance", True, "normal", 0.02, [0]).to(device)

        # Losses
        self.clip_loss = CLIPLoss(device, lambda_direction=args.lambda_direction, lambda_patch=args.lambda_patch, lambda_global=args.lambda_global)
        self.mse_loss  = torch.nn.MSELoss()

        self.source_class = args.source_class
        self.target_class = args.target_class


    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):

        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.style(styles)
            
            frozen_img = self.generator_frozen(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        trainable_img = self.generator_trainable(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        translated_src_to_target = self.cycle_src_to_target(frozen_img)
        translated_target_to_src = self.cycle_target_to_src(trainable_img)

        cycle_loss = self.mse_loss(translated_target_to_src, frozen_img) + self.mse_loss(translated_src_to_target, trainable_img)
        
        clip_loss = self.clip_loss(frozen_img, self.source_class, trainable_img, self.target_class)

        return [frozen_img, trainable_img], [translated_src_to_target, translated_target_to_src], clip_loss, cycle_loss

    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
