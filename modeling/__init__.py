# encoding: utf-8

from .VAE import picVAE


def build_model(cfg):
    model = picVAE(hiddens = cfg.MODEL.HIDDEN_CHNLS, 
                   img_length = cfg.MODEL.IMG_LENGTH, 
                   latent_dim = cfg.MODEL.LATENT_DIM)
    model.to(cfg.DEVICE)
    return model
