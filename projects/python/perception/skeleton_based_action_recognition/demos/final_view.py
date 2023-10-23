
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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from pathlib import Path
from mediapipe.framework.formats import landmark_pb2
import math
from opendr.engine.target import MPPose

# opendr imports
import argparse
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

video_folder_path = '/home/joao/Zed/svo'
svo_path = os.path.join(str(video_folder_path), str('approach_30' + '.svo'))

TARGET_FRAMES = 300
NUM_KEYPOINTS = 24
METHOD = 'tagcn'
MODEL_TO_TEST = 'tagcn_51epochs_0.1lr_150subframes_dropafterepoch3040_batch123_DEARLORD'
#torch1.9.0+cu111
mp_pose = mp.solutions.pose

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
        self.init_params.set_from_svo_file(str(svo_path))
        
        self.rt_param.enable_fill_mode = True    

    def __iter__(self):
        status = self.cam.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)
        return self

    def __next__(self):
        if self.cam.grab(self.rt_param) != sl.ERROR_CODE.SUCCESS:
            raise StopIteration
        
        print('Grabbing frame...')
        if self.cam.grab(self.rt_param) != sl.ERROR_CODE.SUCCESS:
         print("Error grabbing frames.")
         return None, None
        
        left_image = sl.Mat()
        point_cloud = sl.Mat()
        if self.cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) != sl.ERROR_CODE.SUCCESS or \
        self.cam.retrieve_image(left_image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
         print("Error retrieving frame data.")
         return None, None
        
        # Convert the image format and point cloud data
        rgb_image = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB)
        point_cloud_data = point_cloud.get_data()
        
        # Free the memory of the matrices
        left_image.free()
        point_cloud.free()
        
        return rgb_image, point_cloud_data
        return cv2.cvtColor(left_image, cv2.COLOR_BGRA2RGB), point_cloud #returns rgb image to send to mediapipe


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


def pose2numpy(num_frames, poses_list):
    C = 3
    T = TARGET_FRAMES
    V = NUM_KEYPOINTS
    M = 1  # num_person_in
    data_numpy = np.zeros((1, C, num_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))

    for t in range(num_frames):
        data_numpy[0, 0:3, t, :, 0] = np.transpose(poses_list[t].data)

    # if we have less than 75 frames, repeat frames to reach 75
    diff = T - num_frames
    if diff != 0:
        while diff > 0:
            num_tiles = int(diff / num_frames)
            if num_tiles > 0:
                data_numpy = tile(data_numpy, data_numpy.shape[1], num_tiles+1)
                num_frames = data_numpy.shape[2]
                diff = T - num_frames
            elif num_tiles == 0:
                skeleton_seq[:, :, :num_frames, :, :] = data_numpy
                for j in range(diff):
                    skeleton_seq[:, :, num_frames+j, :,
                                 :] = data_numpy[:, :, -1, :, :]
            break
    elif diff == 0:
        skeleton_seq = data_numpy

    return skeleton_seq


def select_2_poses(poses):
    selected_poses = []
    energy = []
    for i in range(len(poses)):
        s = poses[i].data[:, 0].std() + poses[i].data[:, 1].std()
        energy.append(s)
    energy = np.array(energy)
    index = energy.argsort()[::-1][0:2]
    for i in range(len(index)):
        selected_poses.append(poses[index[i]])
    return selected_poses

print()
ACTION_CLASSES = pd.read_csv(os.path.join(Path(__file__).parent,'custom_labels.csv'), verbose=True, index_col=0).to_dict()["name"]


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

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return image

def sort_skeleton_data(results, point_cloud) -> MPPose :
    pose_keypoints = np.ones((NUM_KEYPOINTS, 3), dtype=np.int32) * -1
    try:  # change the z coordinate to get from the generated depth information
        # Left Shoulder
        pose_keypoints[0, 0] = c_x = round(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * 1920)
        pose_keypoints[0, 1] = c_y = round(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * 1080)
        err, point_cloud_value = point_cloud.get_value(c_x, c_y)

        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])

        pose_keypoints[0, 2] = int(distance)
        # keypoints_scores[0, 0] = float(
        #     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility)

        # Right Shoulder
        pose_keypoints[1, 0] = c_x = round(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * 1920)
        pose_keypoints[1, 1] = c_y = round(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[1, 2] = int(distance)
        # keypoints_scores[1, 0] = float(
        #     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)

        # Right Elbow
        pose_keypoints[2, 0] = c_x = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * 1920)
        pose_keypoints[2, 1] = c_y = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[2, 2] = int(distance)
        # keypoints_scores[2, 0] = float(
        #     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility)
    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        skip_frame = True
        
    try:
        for i in range(21):
            # Right Hand
            pose_keypoints[i+3, 0] = c_x = int(
                results.right_hand_landmarks.landmark[i].x * 1920)
            pose_keypoints[i+3, 1] = c_y = int(
                results.right_hand_landmarks.landmark[i].y * 1080)

            err, point_cloud_value = point_cloud.get_value(c_x, c_y)
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
            pose_keypoints[i+3, 2] = int(distance)
            # made up number the hand model does not output visibility scores
            # keypoints_scores[i+3, 0] = float(0.99)
    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        skip_frame = True

    pose = MPPose(pose_keypoints, -1)
    return pose

if __name__ == '__main__':

    # # Action classifier
    # if METHOD == 'tagcn':
    #     action_classifier = SpatioTemporalGCNLearner(device='cpu', dataset_name='custom',
    #                                                  in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=6, num_person=1)
    # else:
    #     print("ERROR! MODEL NOT IMPLEMENTED")
    #     exit()
    #     # action_classifier = SpatioTemporalGCNLearner(device=device, dataset_name='nturgbd_cv', method_name=args.method,
    #     #                                              in_channels=2, num_point=18, graph_type='openpose')

    # print('print_numpoints', action_classifier.num_point)

    # model_saved_path = Path(__file__).parent / 'models' / 'GOOD_MODELS' / str(MODEL_TO_TEST) / 'model'
    # action_classifier.load(model_saved_path, str(MODEL_TO_TEST.split('_DEARLORD')[0]), verbose=True)

    #action_classifier.optimize()

    #image_provider = VideoReader()  # loading a video or get the camera id 0

    counter, avg_fps = 0, 0
    poses_list = []
    window = int(30)
    f_ind = 0
    detector = mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity = 1)
    print('here')
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False  # Convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    rt_param = sl.RuntimeParameters()

    rt_param.enable_fill_mode = True
    
    # Create a Camera object
    zed = sl.Camera()
    
    #Update path to the next svo file
    init_params.set_from_svo_file(f"{svo_path}")
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    
    left_image = sl.Mat()
    
    
    
    while True:
        zed.grab(rt_param)
        
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        
        cv2.imshow("ZED", cv2.resize(left_image.get_data(),
                                   (int(1920 / 1.5), int(1080 / 1.5))))
        
        key = cv2.waitKey(delay=2)
        if key == ord('q'):
            break
    
    for img, point_cloud in image_provider: #returns img in rgb format
        # if f_ind % window == 0:
        #     start_time = time.perf_counter()
        #     img.flags.writeable = False
            
        #     results = detector.process(img)
            
        #     annotated_bgr_image = draw_skeletons(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), results)
        #     pose = sort_skeleton_data(results, point_cloud)
        #     counter += 1
        #     poses_list.append(pose)

        #     if counter > (TARGET_FRAMES): #if more than 150 frames 
        #         poses_list.pop(0)
        #         counter = TARGET_FRAMES
        #     if counter > 0:
        #         skeleton_seq = pose2numpy(counter, poses_list)
        #         prediction = action_classifier.infer(skeleton_seq)
        #         category_labels = preds2label(prediction.confidence)
        #         print(category_labels)
        #         draw_preds(annotated_bgr_image, category_labels)

        #     # Calculate a running average on FPS
        #     end_time = time.perf_counter()
        #     fps = 1.0 / (end_time - start_time)
        #     avg_fps = 0.8 * fps + 0.2 * fps
        #     # Wait a few frames for FPS to stabilize
        #     if counter > 5:
        #         annotated_bgr_image = cv2.putText(annotated_bgr_image, "FPS: %.2f" % (avg_fps,), (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
        #                           1, (255, 0, 0), 2, cv2.LINE_AA)
        #     cv2.imshow('Result', annotated_bgr_image)
        cv2.imshow('yup', img)
        f_ind += 1
        key = cv2.waitKey(500)
        if key == ord('q'):
            break

    print("Average inference fps: ", avg_fps)
    cv2.destroyAllWindows()
