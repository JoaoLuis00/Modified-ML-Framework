import numpy as np
import os
import torch
import cv2
import time
import pandas as pd
from typing import Dict
import pyzed.sl as sl
import mediapipe as mp
from pathlib import Path
import math
from opendr.engine.target import MPPose

# opendr imports
import argparse
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner

TARGET_FRAMES = 250
NUM_KEYPOINTS = 46

MODEL_TO_TEST = 'tagcn_37epochs_0.1lr_100subframes_dropafterepoch5060_batch30'

if MODEL_TO_TEST.split('_')[0] == 'tagcn':
    METHOD = 'tagcn'
elif MODEL_TO_TEST.split('_')[0] == 'stgcn':
    METHOD = 'stgcn'
else:
    METHOD = 'stbln'

ACTION_CLASSES = pd.read_csv(os.path.join(Path(__file__).parent,'custom_labels.csv'), verbose=True, index_col=0).to_dict()["name"]


def preds2label(confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    labels = {ACTION_CLASSES[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
    return labels


action_classifier = SpatioTemporalGCNLearner(device='cuda', dataset_name='custom', method_name=METHOD, num_frames=TARGET_FRAMES,
                                            in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=6, num_person=1)

print('print_numpoints', action_classifier.num_point)

model_saved_path = Path(__file__).parent / 'models' / 'final_atualizado_fullsize' / str(MODEL_TO_TEST) / 'model'
action_classifier.load(model_saved_path, MODEL_TO_TEST, verbose=True)

load_data = np.load(str(Path(__file__).parent / 'data' / "final_atualizado_fullsize/val_joints.npy"), allow_pickle=True)

one_sample = load_data[0,...]

def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    tiled_a = torch.index_select(a, dim, order_index)
    return tiled_a.numpy()

num_tiles = int(150 / 150)

sample_npy = np.zeros((1,3,TARGET_FRAMES,46,1))

sample_npy[0,:,:,:,:] = one_sample
list_=[]
contamos=0
for i in range(TARGET_FRAMES):
    sample = load_data[0,:,i,...]
    list_.append(sample)
    sample_npy = np.zeros((1,3,TARGET_FRAMES,46,1)) * -1

    for t in range(i):
         sample_npy[0, :, t, :, :] = (list_[t])

    num_tiles = int(TARGET_FRAMES - (i+1) / (i+1))
    #sample_npy[0,:,0,:,:] = sample
    #num_tiles = int(300 - (1) / (1))

    one_sample = tile(sample_npy, 2, num_tiles+1)

    prediction = action_classifier.infer(sample_npy)

    category_labels = preds2label(prediction.confidence)

    print(category_labels)
    

#one_sample = tile(sample_npy, 2, num_tiles+1)

prediction = action_classifier.infer(sample_npy)

category_labels = preds2label(prediction.confidence)

first_key = next(iter(category_labels))
first_value = category_labels[first_key]

new_key = 'approachamos'

#category_labels = {new_key: first_value, **{k: v for k,v in category_labels.items() if k!=new_key and k!= first_key}}

print(category_labels)

