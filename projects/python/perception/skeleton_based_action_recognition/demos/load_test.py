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

TARGET_FRAMES = 300
NUM_KEYPOINTS = 24
METHOD = 'tagcn'
#MODEL_TO_TEST = 'stgcn_37epochs_0.1lr_100subframes_dropafterepoch5060_batch30'
#MODEL_TO_TEST = 'tagcn_35epochs_0.1lr_100subframes_dropafterepoch5060_batch15'
MODEL_TO_TEST = 'tagcn_54epochs_0.1lr_125subframes_dropafterepoch5060_batch15'

ACTION_CLASSES = pd.read_csv(os.path.join(Path(__file__).parent,'custom_labels.csv'), verbose=True, index_col=0).to_dict()["name"]


def preds2label(confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    labels = {ACTION_CLASSES[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
    return labels


action_classifier = SpatioTemporalGCNLearner(device='cpu', dataset_name='custom', method_name=METHOD,
                                            in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=4, num_person=1)

print('print_numpoints', action_classifier.num_point)

model_saved_path = Path(__file__).parent / 'models' / 'final_v2' / str(MODEL_TO_TEST) / 'model'
action_classifier.load(model_saved_path, MODEL_TO_TEST, verbose=True)

load_data = np.load(str(Path(__file__).parent / 'data' / "final_v2/val_joints.npy"), allow_pickle=True)

one_sample = load_data[0,...]

sample_npy = np.zeros((1,3,300,24,1))

sample_npy[0,:,:,:,:] = one_sample

prediction = action_classifier.infer(sample_npy)

category_labels = preds2label(prediction.confidence)
print(category_labels)

