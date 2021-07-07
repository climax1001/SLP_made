# coding: utf-8
"""
Defining global constants
"""

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
dirname = '/home/juncislab/Downloads/bbinix/' \
          'PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
train_dir = '/features/fullFrame-210x260px/train'
TARGET_PAD = 0.0

WANNA_POSE = ["LEFT_ELBOW_X","LEFT_ELBOW_Y","RIGHT_ELBOW_X","RIGHT_ELBOW_Y","LEFT_SHOULDER_X","LEFT_SHOULDER_Y","RIGHT_SHOULDER_X","RIGHT_SHOULDER_Y"]


DEFAULT_UNK_ID = lambda: 0
