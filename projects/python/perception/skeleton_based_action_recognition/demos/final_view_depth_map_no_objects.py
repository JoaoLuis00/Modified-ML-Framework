
# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
from math import cos, sin

# opendr imports
#from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

video_folder_path = '/home/joao/Zed'
svo_path = os.path.join(str(video_folder_path), str('uncompressed' + '/request_40' + '.svo'))

TARGET_FRAMES = 250
NUM_KEYPOINTS = 46
ORIGIN = (0,0,0)
WRIST = mp.solutions.holistic.HandLandmark.WRIST
LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW

RECORD = False

DATA_TYPE = 'depth_map_2'

# LR 0.1
MODEL_TO_TEST = 'tagcn_50epochs_0.1lr_50subframes_dropafterepoch3040_batch64'
MODEL_TO_TEST = 'tagcn_33epochs_0.1lr_30subframes_dropafterepoch3040_batch64'
#MODEL_TO_TEST = 'tagcn_50epochs_0.01lr_50subframes_dropafterepoch3040_batch64'


#MODEL_TO_TEST = 'stgcn_28epochs_0.01lr_dropafterepoch3040_batch30'


if MODEL_TO_TEST.split('_')[0] == 'tagcn':
    METHOD = 'tagcn'
else:
    METHOD = 'stgcn'
#torch1.9.0+cu111

ACTION_CLASSES = pd.read_csv(os.path.join(Path(__file__).parent,'custom_labels.csv'), verbose=True, index_col=0).to_dict()["name"]

class VideoReader(object):
    
        # #Create a InitParameters object and set configuration parameters
        # init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        # init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        # init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        # init_params.camera_fps = 30
    
    def __init__(self):
        '''SETUP ZED PARAMETERS'''
        # Start ZED OBJECT for camera
        self.cam = sl.Camera()
        self.init_params = sl.InitParameters()
        self.rt_param = sl.RuntimeParameters()
       
        self.init_params.svo_real_time_mode = False  # Convert in realtime
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER     # Set coordinate units
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        #self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        self.init_params.set_from_svo_file(str(svo_path))
        
        self.rt_param.enable_fill_mode = True    

    def __iter__(self):
        status = self.cam.open(self.init_params)
        #self.cam.set_svo_position(10)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)
        return self

    def __next__(self):
        #print(self.cam.get_svo_position())
        if self.cam.grab(self.rt_param) != sl.ERROR_CODE.SUCCESS:
            raise StopIteration
        
        left_image = sl.Mat()
        depth_map = sl.Mat()
        if self.cam.retrieve_measure(depth_map, sl.MEASURE.DEPTH) != sl.ERROR_CODE.SUCCESS or \
        self.cam.retrieve_image(left_image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
         print("Error retrieving frame data.")
         raise StopIteration
        # Convert the image format and point cloud data
        rgb_image = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB)
        return rgb_image, depth_map

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

def pose2numpy(num_frames, poses_list, num_channels=4):
    C = num_channels
    T = TARGET_FRAMES
    V = NUM_KEYPOINTS
    M = 1  # num_person_in
    data_numpy = np.zeros((1, C, num_frames, V, M))
    skeleton_seq = np.ones((1, C, T, V, M)) * -1

    for t in range(num_frames):
        data_numpy[0, 0:3, t, :, 0] = np.transpose(poses_list[t].data)

    diff = T - num_frames
    if diff == 0:
        skeleton_seq = data_numpy
    while diff > 0:
        num_tiles = int(diff / num_frames)
        if num_tiles > 0:
            data_numpy = tile(data_numpy, 2, num_tiles+1)
            num_frames = data_numpy.shape[2]
            diff = T - num_frames
        elif num_tiles == 0:
            skeleton_seq[:, :, :num_frames, :, :] = data_numpy
            for j in range(diff):
                skeleton_seq[:, :, num_frames+j, :, :] = data_numpy[:, :, -1, :, :]
            break
    
    return skeleton_seq

def preds2label(confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    labels = {ACTION_CLASSES[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
    return labels

def draw_preds(frame, preds: Dict):
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(frame, f"{prob:04.3f} {cls}",
                    (10, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2,)
        
def draw_skeletons(image, results):
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return image

def sort_skeleton_data(results, depth_map):
    pose_keypoints = np.ones((NUM_KEYPOINTS, 3), dtype=np.int32) * -1
    missed_hand_left = missed_hand_right = missed_pose = False

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
    
        try:  # change the z coordinate to get from the generated depth information
            # Left Shoulder
            pose_keypoints[0, 0] = c_x = round(pose_landmarks[LEFT_SHOULDER].x * 1920)
            pose_keypoints[0, 1] = c_y = round(pose_landmarks[LEFT_SHOULDER].y * 1080)
            err, depth_val = depth_map.get_value(c_x, c_y)
            if err == sl.ERROR_CODE.SUCCESS:
                pose_keypoints[0, 2] = round(depth_val)
            else:
                missed_pose = True
        except (IndexError, AttributeError) as e:
            missed_pose = True

        try:
            # Right Shoulder
            pose_keypoints[1, 0] = c_x = round(pose_landmarks[RIGHT_SHOULDER].x * 1920)
            pose_keypoints[1, 1] = c_y = round(pose_landmarks[RIGHT_SHOULDER].y * 1080)
            err, depth_val = depth_map.get_value(c_x, c_y)
            if err == sl.ERROR_CODE.SUCCESS:
                pose_keypoints[1, 2] = round(depth_val)
            else:
                missed_pose = True
        except (IndexError, AttributeError) as e:
            missed_pose = True
            
        try:
            # Left Elbow
            pose_keypoints[2, 0] = c_x = round(pose_landmarks[LEFT_ELBOW].x * 1920)
            pose_keypoints[2, 1] = c_y = round(pose_landmarks[LEFT_ELBOW].y * 1080)
            err, depth_val = depth_map.get_value(c_x, c_y)
            if err == sl.ERROR_CODE.SUCCESS:
                pose_keypoints[2, 2] = round(depth_val)
            else:
                missed_pose = True
        except (IndexError, AttributeError) as e:
            missed_pose = True
            
        try:
            # Right Elbow
            pose_keypoints[3, 0] = c_x = round(pose_landmarks[RIGHT_ELBOW].x * 1920)
            pose_keypoints[3, 1] = c_y = round(pose_landmarks[RIGHT_ELBOW].y * 1080)
            err, depth_val = depth_map.get_value(c_x, c_y)
            if err == sl.ERROR_CODE.SUCCESS:
                pose_keypoints[3, 2] = round(depth_val)
            else:
                missed_pose = True
        except (IndexError, AttributeError) as e:
            missed_pose = True
        
    if not results.left_hand_landmarks:
        missed_hand_left = True
    else:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        #Left Wrist
        pose_keypoints[4, 0] = c_x = round(left_hand_landmarks[WRIST].x * 1920)
        pose_keypoints[4, 1] = c_y = round(left_hand_landmarks[WRIST].y * 1080)
        err, depth_val = depth_map.get_value(c_x, c_y)
        if err == sl.ERROR_CODE.SUCCESS:
            d_lwrist = depth_val
            pose_keypoints[4, 2] = round(d_lwrist)
            for idx, landmark in enumerate(left_hand_landmarks[1:]):
                x, y, z = landmark.x * 1920, landmark.y * 1080, d_lwrist + d_lwrist*landmark.z
                pose_keypoints[idx+5, 0] = round(x)
                pose_keypoints[idx+5, 1] = round(y)
                pose_keypoints[idx+5, 2] = round(z) 
        else:
            missed_hand_left = True

    if not results.right_hand_landmarks:
        missed_hand_right = True
    else:
        right_hand_landmarks =  results.right_hand_landmarks.landmark
        #Right Wrist
        pose_keypoints[25, 0] = c_x = round(right_hand_landmarks[WRIST].x * 1920)
        pose_keypoints[25, 1] = c_y = round(right_hand_landmarks[WRIST].y * 1080)
        err, depth_val = depth_map.get_value(c_x, c_y)
        if err == sl.ERROR_CODE.SUCCESS:
            d_rwrist = depth_val
            pose_keypoints[25, 2] = round(d_rwrist)
            for idx, landmark in enumerate(right_hand_landmarks[1:]):
                x, y, z = landmark.x * 1920, landmark.y * 1080, d_rwrist + d_rwrist*landmark.z
                pose_keypoints[idx+26, 0] = round(x)
                pose_keypoints[idx+26, 1] = round(y)
                pose_keypoints[idx+26, 2] = round(z)
        else:
            missed_hand_right = True

    pose = MPPose(pose_keypoints, -1)

    return pose, missed_pose, missed_hand_left, missed_hand_right      
         
if __name__ == '__main__':
    
    action = 'action'
    output_path = Path(f'/home/joao/Zed/{MODEL_TO_TEST}_{action}.avi')
    if RECORD:
        video_writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc('M', '4', 'S', '2'),30,(1920, 1080)) 

    # Action classifier
    action_classifier = SpatioTemporalGCNLearner(device='cpu', dataset_name='custom', method_name=METHOD, num_frames=TARGET_FRAMES,
                                                 in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=6, num_person=1)

    model_saved_path = Path(__file__).parent / 'models' / str(DATA_TYPE) / str(MODEL_TO_TEST) / 'model'
    action_classifier.load(model_saved_path, MODEL_TO_TEST, verbose=True)

    #action_classifier.optimize()

    image_provider = VideoReader()

    counter, avg_fps = 0, 0
    poses_list = []
    window = 1
    f_ind = 0
    detector = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity = 1)
    time.sleep((5))

    list_contours = []
    missed_hand = False
    left_hand_center_point_old = None
    right_hand_center_point_old = None
    
    for img_rgb, depth_map in image_provider:
        if f_ind % window == 0:
            start_time = time.perf_counter()

            img_rgb.flags.writeable = False
            results = detector.process(img_rgb)
            img_rgb.flags.writeable = True

            pose, missed_pose, missed_hand_left, missed_hand_right = sort_skeleton_data(results, depth_map)
            if not missed_pose and not missed_hand_left and not missed_hand_left:
                counter += 1
                poses_list.append(pose)
            
            if counter > int(TARGET_FRAMES): #if more than x frames 
                poses_list.pop(0)
                counter = int(TARGET_FRAMES)
            if counter > 0:
                skeleton_seq = pose2numpy(counter, poses_list,3)
                prediction = action_classifier.infer(skeleton_seq)
                category_labels = preds2label(prediction.confidence)
                print(category_labels)

                annotated_bgr_image = draw_skeletons(img_rgb, results)
                draw_preds(annotated_bgr_image, category_labels)
            
                # # Calculate a running average on FPS
                end_time = time.perf_counter()
                fps = 1.0 / (end_time - start_time)
                avg_fps = 0.8 * fps + 0.2 * fps
                print(counter)
                if counter > 5:
                    annotated_bgr_image = cv2.putText(annotated_bgr_image, "FPS: %.2f" % (avg_fps,), (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                if RECORD:
                    video_writer.write(annotated_bgr_image)
                cv2.imshow('Result', annotated_bgr_image)
        f_ind += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    #print("Average inference fps: ", avg_fps)
    cv2.destroyAllWindows()
