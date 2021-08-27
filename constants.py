# coding: utf-8
"""
Defining global constants
"""
from helpers import Config

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
dirname = '/home/juncislab/Downloads/bbinix/' \
          'PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
train_dir = '/features/fullFrame-210x260px/test'
TARGET_PAD = 0.0

WANNA_POSE = ["LEFT_ELBOW_X","LEFT_ELBOW_Y","RIGHT_ELBOW_X","RIGHT_ELBOW_Y","LEFT_SHOULDER_X","LEFT_SHOULDER_Y","RIGHT_SHOULDER_X","RIGHT_SHOULDER_Y"]

config = Config({
    "n_enc_vocab": 7096,
    "n_dec_vocab": 7096,
    "n_enc_seq": 256,
    "n_dec_seq": 256,
    "n_layer": 6,
    "d_hidn": 256,
    "i_pad": 0,
    "d_ff": 1024,
    "n_head": 4,
    "d_head": 64,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-12
})
DEFAULT_UNK_ID = lambda: 0
