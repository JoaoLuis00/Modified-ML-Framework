
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

TARGET_FRAMES = 300
num_keypoints = 46
mp_pose = mp.solutions.pose

class VideoReader(object):
    def __init__(self, file_name):
        '''SETUP ZED PARAMETERS'''
        # Start ZED OBJECT for camera
        cam = sl.Camera()
        
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init_params.camera_fps = 30

        rt_param = sl.RuntimeParameters()

        rt_param.enable_fill_mode = True    

    def __iter__(self):

        status = self.cam.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit(1)
        return self

    def __next__(self):
        if self.cam.grab(self.rt_param) != sl.ERROR_CODE.SUCCESS:
            raise StopIteration
        left_image = sl.Mat()
        point_cloud = sl.Mat()
        self.cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        self.cam.retrieve_image(left_image, sl.VIEW.LEFT)
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
    V = num_keypoints
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


ACTION_CLASSES = pd.read_csv("./custom_labels.csv", verbose=True, index_col=0).to_dict()["name"]


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
        
def draw_skeletons(img):
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_results.pose_landmarks[0]])
    mp.solutions.drawing_utils.draw_landmarks(
        img,
        pose_landmarks_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    hand_landmarks_list = hands_results.hand_landmarks
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            img,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
    return img

def sort_skeleton_data(pose_results, hands_results, point_cloud):
    pose_keypoints = np.ones((num_keypoints, 3), dtype=np.int32) * -1
    keypoints_scores = np.ones(
                (num_keypoints, 1), dtype=np.float32) * -1
    try:  # change the z coordinate to get from the generated depth information
        # Left Shoulder
        pose_keypoints[0, 0] = c_x = round(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_SHOULDER].x * 1920)
        pose_keypoints[0, 1] = c_y = round(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_SHOULDER].y * 1080)
        err, point_cloud_value = point_cloud.get_value(c_x, c_y)

        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])

        pose_keypoints[0, 2] = int(distance)
        #keypoints_scores[0, 0] = float(
        #    pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_SHOULDER].visibility)
    except IndexError:
        pass
    
    try:
        # Right Shoulder
        pose_keypoints[1, 0] = c_x = round(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_SHOULDER].x * 1920)
        pose_keypoints[1, 1] = c_y = round(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_SHOULDER].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[1, 2] = int(distance)
        #keypoints_scores[1, 0] = float(
        #    pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility)
    except IndexError:
        pass
    
    try:
        # Left Elbow
        pose_keypoints[2, 0] = c_x = int(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_ELBOW].x * 1920)
        pose_keypoints[2, 1] = c_y = int(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_ELBOW].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])

        pose_keypoints[2, 2] = int(distance)
        # keypoints_scores[2, 0] = float(
        #     pose_results.pose_landmarks[0][mp_pose.PoseLandmark.LEFT_ELBOW].visibility)
    except IndexError:
        pass
    
    try:
        # Right Elbow
        pose_keypoints[3, 0] = c_x = int(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_ELBOW].x * 1920)
        pose_keypoints[3, 1] = c_y = int(
            pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_ELBOW].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[3, 2] = int(distance)
        #keypoints_scores[3, 0] = float(
            #pose_results.pose_landmarks[0][mp_pose.PoseLandmark.RIGHT_ELBOW].visibility)
    except IndexError:
        pass

    for i in range(21):
        # Left Hand
        try:
            pose_keypoints[i+4, 0] = c_x = int(
                hands_results.hand_landmarks[0][i].x * 1920)
            pose_keypoints[i+4, 1] = c_y = int(
                hands_results.hand_landmarks[0][i].y * 1080)

            err, point_cloud_value = point_cloud.get_value(c_x, c_y)
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])
            pose_keypoints[i+4, 2] = int(distance)
            # made up number the hand model does not output visibility scores
            #keypoints_scores[i+4, 0] = float(0.99)
        except IndexError:
            pass

        try:
            # Right Hand
            pose_keypoints[i+25, 0] = c_x = int(
                hands_results.hand_landmarks[1][i].x * 1920)
            pose_keypoints[i+25, 1] = c_y = int(
                hands_results.hand_landmarks[1][i].y * 1080)

            err, point_cloud_value = point_cloud.get_value(c_x, c_y)
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])
            pose_keypoints[i+25, 2] = int(distance)
            # made up number the hand model does not output visibility scores
            #keypoints_scores[i+25, 0] = float(0.99)
        except IndexError:
            pass

    return MPPose(pose_keypoints, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu")
    parser.add_argument('--method', type=str, default='tagcn',
                        help='action detection method')
    parser.add_argument('--action_checkpoint_name', type=str, default='stgcn_ntu_cv_lw_openpose',
                        help='action detector model name')
    parser.add_argument('--num_frames', type=int, default=300,
                        help='number of frames to be processed for each action')
    parser.add_argument('--fps', type=int, default=30,
                        help='number of frames per second to be processed by pose estimator and action detector')

    args = parser.parse_args()
    onnx, device = args.onnx, args.device
    accelerate = args.accelerate
    onnx, device, accelerate = args.onnx, args.device, args.accelerate

    # pose estimator
    VisionRunningMode = mp.tasks.vision.RunningMode

    pose_landmarker_base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    hand_landmarker_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

    #! TEST FOR LIVESTREAM IF NOT IMPLEMENT FOR SINGLE IMAGES
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions(
                            base_options=pose_landmarker_base_options,
                            running_mode=VisionRunningMode.VIDEO,
                            min_pose_detection_confidence=0.9,
                            min_tracking_confidence=0.9)


    HandLandmarkerOptions = vision.HandLandmarkerOptions(base_options=hand_landmarker_base_options,
                                        num_hands=2,
                                        running_mode=VisionRunningMode.VIDEO,
                                        min_hand_detection_confidence=0.4,
                                        min_tracking_confidence=0.6)
    
    pose_detector = vision.PoseLandmarker.create_from_options(PoseLandmarkerOptions)
    hand_detector = vision.HandLandmarker.create_from_options(HandLandmarkerOptions)
    # Action classifier
    if args.method == 'tagcn':
        action_classifier = SpatioTemporalGCNLearner(device=device, dataset_name='custom',
                                                     in_channels=3,num_point=46, graph_type='custom', num_class=5, num_person=1)
    else:
        print("ERROR MODEL NOT IMPLEMENTED")
        exit()
        # action_classifier = SpatioTemporalGCNLearner(device=device, dataset_name='nturgbd_cv', method_name=args.method,
        #                                              in_channels=2, num_point=18, graph_type='openpose')

    print('print_numpoints', action_classifier.num_point)

    model_saved_path = Path(__file__).parent / 'models'
    action_classifier.load(model_saved_path, 'CHANGE NAME TO WHATEVER ONE WANTS')

    # Optimization
    if onnx:
        action_classifier.optimize()

    image_provider = VideoReader(args.video)  # loading a video or get the camera id 0

    counter, avg_fps = 0, 0
    poses_list = []
    window = int(30/args.fps)
    f_ind = 0
    for img, point_cloud in image_provider:
        if f_ind % window == 0:
            start_time = time.perf_counter()
            frame_to_process = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            pose_results = pose_detector.detect_for_video(frame_to_process)
            hands_results = hand_detector.detect_for_video(frame_to_process)
            
            if pose_results.pose_landmarks:
                annotated_bgr_image = draw_skeletons(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pose_results, hands_results)
                pose = sort_skeleton_data(pose_results, hands_results, point_cloud)
                counter += 1
                poses_list.append(pose)

            if counter > args.num_frames:
                poses_list.pop(0)
                counter = args.num_frames
            if counter > 0:
                skeleton_seq = pose2numpy(counter, poses_list)

                prediction = action_classifier.infer(skeleton_seq)
                category_labels = preds2label(prediction.confidence)
                print(category_labels)
                draw_preds(img, category_labels)

            # Calculate a running average on FPS
            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)
            avg_fps = 0.8 * fps + 0.2 * fps
            # Wait a few frames for FPS to stabilize
            if counter > 5:
                img = cv2.putText(img, "FPS: %.2f" % (avg_fps,), (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Result', img)
        f_ind += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print("Average inference fps: ", avg_fps)
    cv2.destroyAllWindows()
