"""
Convert TVQA into tfrecords
"""
import sys

sys.path.append('../../')
import argparse
import hashlib
import io
import json
import os
import random
import numpy as np
from tempfile import TemporaryDirectory
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from google.cloud import storage
# from sacremoses import MosesDetokenizer
import regex as re
from tqdm import tqdm
import pandas as pd
from finetune.common_data_utils import *
from collections import defaultdict
import colorsys
import hashlib
import tempfile
import subprocess
from scipy.io import wavfile
from mreserve.preprocess import make_spectrogram, invert_spectrogram
from mreserve.lowercase_encoder import START
# import pysrt
# from unidecode import unidecode
import ftfy
import time
import yaml


parser = create_base_parser()
parser.add_argument(
    '-data_dir',
    dest='data_dir',
    default='/mnt/disks/persist/merlot_reserve/finetune/datasets/nextqa',
    type=str,
    help='Image directory.'
)
parser.add_argument(
    '-next_qa_file',
    dest='next_qa_file',
    default='txt_nextqa_blip_clip_loss.json',
    type=str,
    help='NextQA version'
)
parser.add_argument(
    'pretrain_config_file',
    help='Where the config.yaml is located',
    type=str
)
parser.add_argument(
    'split',
    help='split dataset',
    type=str
)
args = parser.parse_args()
random.seed(args.seed)
pretrain_config_file = '/mnt/disks/persist/merlot_reserve/finetune/nextqa/base.yaml'

with open(pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

config['num_answers'] = 5
config['num_segments'] = 7

data_fn = os.path.join(args.data_dir, 'subtitles', args.next_qa_file)
with open(data_fn, 'r') as f:
    for idx, l in enumerate(f):
        # __import__('ipdb').set_trace()
        data = json.loads(l)[args.split]

def parse_item(item):
    qa_item = {'qa_query': item.pop('question'), 'qa_choices': [item.pop(f'option_{i}') for i in range(5)],
               'qa_label': item.get('answer', 0),
               'id':  item['video']}

    frames_path = os.path.join(args.data_dir, 'all_frames_whole', f"{item['video']}.mp4")
    # frames_path = os.path.join(args.data_dir, 'all_frames', item['vid_name'])
    frames = []
    for fn in os.listdir(frames_path):
        image = Image.open(os.path.join(frames_path, fn))
        image = resize_image(image, shorter_size_trg=450, longer_size_max=800)
        frames.append(image)

    # Figure out the relative position of the annotation
    qa_item['num_frames'] = len(frames)
    qa_item['_frames_path'] = frames_path

    # Pad to 7
    for i in range(7 - len(frames)):
        frames.append(frames[-1])
    return qa_item, frames

num_written = 0
max_len = 0

for fold in range(args.num_folds):
    print(f"\n============================Fold {fold}============================")
    start = time.time()

    out_fn = os.path.join(args.data_dir, 'tfrecord', '{}{:03d}of{:03d}.tfrecord'.format(args.split, fold, args.num_folds))
    with GCSTFRecordWriter(out_fn, auto_close=False) as tfrecord_writer:
        while True:
            item = data[num_written]
            qa_item, frames = parse_item(item)

            # Tack on the relative position of the localized timestamp, plus a START token for separation
            query_enc = encoder.encode(qa_item['qa_query']).ids
            feature_dict = {
                'id': bytes_feature(str(qa_item['id']).encode('utf-8')),
                'qa_query': int64_list_feature(query_enc),
                'qa_label': int64_feature(qa_item['qa_label']),
                'num_frames': int64_feature(qa_item['num_frames']),
            }
            max_query = 0
            for j, choice_j in enumerate(encoder.encode_batch(qa_item['qa_choices'])):
                feature_dict[f'qa_choice_{j}'] = int64_list_feature(choice_j.ids)
                max_query = max(len(choice_j.ids) + len(query_enc), max_query)
            for i, frame_i in enumerate(frames):
                feature_dict[f'c{i:02d}/image_encoded'] = bytes_feature(pil_image_to_jpgstring(frame_i))

            max_len = max(max_len, max_query)

            # if num_written < 4:
            print(f"~~~~~~~~~~~ Example {num_written} {qa_item['id']} ~~~~~~~~")
            print(encoder.decode(feature_dict['qa_query'].int64_list.value, skip_special_tokens=False), flush=True)
            for k in range(config['num_answers']):
                toks = feature_dict[f'qa_choice_{k}'].int64_list.value
                toks_dec = encoder.decode(toks, skip_special_tokens=False)
                lab = ' GT' if k == qa_item['qa_label'] else '   '
                print(f'{k}{lab}) {toks_dec}     ({len(toks)}tok)', flush=True)
            
            # Debug image
            os.makedirs('debug', exist_ok=True)
            for i in range(7):
                with open(f'debug/ex{num_written}_img{i}.jpg', 'wb') as f:
                    f.write(feature_dict[f'c{i:02d}/image_encoded'].bytes_list.value[0])
            
            frames_path = qa_item['_frames_path']
            os.system(f'cp -r {frames_path} debug/ex{num_written}_frames')
            # assert False
                            
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            tfrecord_writer.write(example.SerializeToString())
            num_written += 1
            if num_written % 100 == 0:
                print("Have written {} / {}".format(num_written, len(data)), flush=True)
            if num_written % (len(data)//args.num_folds) == 0:
                break
        end = time.time()
        print(f">>>>> Time to write data to fold {fold}: {end-start}")
        tfrecord_writer.close()
    # break

print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)


# ~~~~~~~~~~~ Example 0 2399344595 ~~~~~~~~
#  a man pouring a glass of water into a small child. where is this happening
# 0   )  karaoke room     (2tok)
# 1   )  hiking trail     (2tok)
# 2 GT)  kitchen     (1tok)
# 3   )  cycling trail     (2tok)
# 4   )  room     (1tok)
# ~~~~~~~~~~~ Example 1 2399344595 ~~~~~~~~
#  a young boy pouring a bottle of water into a child's hand. why did the man invert the bottle
# 0   )  feel the music     (3tok)
# 1   )  blow air into it     (4tok)
# 2   )  talking to surrounding people     (4tok)
# 3   )  talking to baby     (3tok)
# 4 GT)  for water to drip down     (5tok)
# ~~~~~~~~~~~ Example 2 2399344595 ~~~~~~~~
#  a man and a child are in a kitchen with a bottle of water. what does the boy do after receiving the red bottle at the start
# 0   )  move backwards     (2tok)
# 1   )  look at that direction     (4tok)
# 2   )  keep chewing food     (3tok)
# 3   )  swims     (1tok)
# 4 GT)  put at the side to dry     (6tok)
# ~~~~~~~~~~~ Example 3 2399344595 ~~~~~~~~
#  a man in a kitchen washing a bowl of fruit. how did the boy get the man to play with him in the water
# 0   )  dribble the ball     (3tok)
# 1   )  play with it     (3tok)
# 2   )  kiss it     (2tok)
# 3   )  slide down     (2tok)
# 4 GT)  put man s hand into water     (6tok)
# Finished writing 11 questions; max len = 36
