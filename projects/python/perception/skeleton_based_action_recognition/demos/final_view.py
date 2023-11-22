
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
from math import atan2, cos, sin, sqrt, pi

# opendr imports
import argparse
from opendr.perception.skeleton_based_action_recognition import ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

video_folder_path = '/home/joao/Zed'
svo_path = os.path.join(str(video_folder_path), str('uncompressed' + '/approach_10' + '.svo'))

TARGET_FRAMES = 200
NUM_KEYPOINTS = 24
#MODEL_TO_TEST = 'stgcn_37epochs_0.1lr_100subframes_dropafterepoch5060_batch30'
#MODEL_TO_TEST = 'tagcn_35epochs_0.1lr_100subframes_dropafterepoch5060_batch15'
#MODEL_TO_TEST = 'tagcn_54epochs_0.1lr_125subframes_dropafterepoch5060_batch15'
#MODEL_TO_TEST = 'tagcn_23epochs_0.1lr_150subframes_dropafterepoch5060_batch10'
#MODEL_TO_TEST = 'tagcn_52epochs_0.1lr_175subframes_dropafterepoch5060_batch15'
#MODEL_TO_TEST = 'tagcn_70epochs_0.1lr_100subframes_dropafterepoch5060_batch61'


#MODEL_TO_TEST = 'tagcn_50epochs_0.1lr_100subframes_dropafterepoch5060_batch15'
MODEL_TO_TEST = 'tagcn_50epochs_0.1lr_75subframes_dropafterepoch3040_batch15'

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
        if self.cam.grab(self.rt_param) != sl.ERROR_CODE.SUCCESS:
            raise StopIteration
        
        left_image = sl.Mat()
        point_cloud = sl.Mat()
        if self.cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA) != sl.ERROR_CODE.SUCCESS or \
        self.cam.retrieve_image(left_image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
         print("Error retrieving frame data.")
         raise StopIteration
        
        # Convert the image format and point cloud data
        rgb_image = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2RGB)
        #point_cloud_data = point_cloud
        
        # # Free the memory of the matrices
        # left_image.free()
        # point_cloud.free()
        
        return rgb_image, point_cloud
        #return cv2.cvtColor(left_image, cv2.COLOR_BGRA2RGB), point_cloud #returns rgb image to send to mediapipe

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
        
def draw_skeletons(image, results):#, contours, obj_center_point, obj_edge_point):
    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # try:
    #     image = drawContours(image, contours, center=obj_center_point, top_point=obj_edge_point)
    # except (IndexError,TypeError):
    #     pass
    
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

def sort_skeleton_data(results, point_cloud):
    pose_keypoints = np.ones((NUM_KEYPOINTS, 3), dtype=np.int32) * -1
    hand_keypoints_list = []
    missed_hand = missed_pose = False
    mean_x = mean_y = mean_z = -1

    try:  # change the z coordinate to get from the generated depth information
        # Left Shoulder
        pose_keypoints[0, 0] = c_x = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * 1920)
        pose_keypoints[0, 1] = c_y = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * 1080)
        err, point_cloud_value = point_cloud.get_value(c_x, c_y)

        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])

        pose_keypoints[0, 2] = int(distance)

    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        missed_pose = True

    try:
        # Right Shoulder
        pose_keypoints[1, 0] = c_x = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * 1920)
        pose_keypoints[1, 1] = c_y = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[1, 2] = int(distance)

    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        missed_pose = True

    try:    
        # Right Elbow
        pose_keypoints[2, 0] = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * 1920)
        pose_keypoints[2, 1] = int(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * 1080)

        err, point_cloud_value = point_cloud.get_value(c_x, c_y)
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        pose_keypoints[2, 2] = int(distance)

    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        missed_pose = True
        
    try:
        for i in range(21):
            # Right Hand
            pose_keypoints[i+3, 0] = int(
                results.right_hand_landmarks.landmark[i].x * 1920)
            pose_keypoints[i+3, 1] = int(
                results.right_hand_landmarks.landmark[i].y * 1080)

            err, point_cloud_value = point_cloud.get_value(c_x, c_y)
            distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
            pose_keypoints[i+3, 2] = int(distance)
            c_z = int(distance)
            
            hand_keypoints_list.append((c_x,c_y,c_z))

    except (IndexError, AttributeError) as e:
        # print(f"Values missing Pose -> sample: {s_name}")
        missed_hand = True

    pose = MPPose(pose_keypoints, -1)

    if not missed_hand:
        mean_x = sum(coord[0] for coord in hand_keypoints_list) / len(hand_keypoints_list)
        mean_y = sum(coord[1] for coord in hand_keypoints_list) / len(hand_keypoints_list)
        mean_z = sum(coord[2] for coord in hand_keypoints_list) / len(hand_keypoints_list)

    # Center point or midpoint of the hand in 3D space
    hand_center_point = (mean_x, mean_y, mean_z)

    return pose, hand_center_point,missed_pose, missed_hand

def getOrientation(img):

  img_spliced = img[500:1000,200:1650]
  #cv22.imshow('Input Image', img)

  # Convert image to grayscale
  gray = cv2.cvtColor(img_spliced, cv2.COLOR_BGR2GRAY)
  #cv22.imshow('Image', gray)
  
  # Convert image to binary
  _, bw = thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #cv22.threshold(gray, 20, 200, cv22.THRESH_TOZERO_INV)#
 
  # Find all the contours in the thresholded image
  contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, offset=(200,500))
  no_contours_detected = False
  filtered_countours = []
  for i, c in enumerate(contours):
 
    # Calculate the area of each contour
    area = cv2.contourArea(c)
   # Ignore contours that are too small or too large
    rect= cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    center = (int(rect[0][0]),int(rect[0][1])) 
    width = int(rect[1][0])
    height = int(rect[1][1])

    x,y = center[0], center[1]

    # Ignore contours that are too small or too large
    if area < 4000 or 100000 < area or x < 400 or x > 1500 or y > 750 or y < 580 or width > 100:
      continue
    else:
      print(width)
      filtered_countours.append(c)
      #print(x,y)
  #print(len(filtered_countours))
  if len(filtered_countours) == 0:
     no_contours_detected = True
     angle = center = top_point = -1
     return angle, center, top_point, filtered_countours, no_contours_detected

  rect = cv2.minAreaRect(filtered_countours[0])

  # Retrieve the key parameters of the rotated bounding box
  center = (int(rect[0][0]),int(rect[0][1])) 
  width = int(rect[1][0])
  height = int(rect[1][1])
  center_x = center[0]
  center_y = center[1]
  half_width = width / 2
  half_height = height / 2
  angle = int(rect[2])
  angle_deg = np.deg2rad(angle)

  # Calculate coordinates of the end points
  x1 = (center_x - half_width * cos(angle_deg) - half_height * sin(angle_deg))
  y1 = (center_y - half_width * sin(angle_deg) + half_height * cos(angle_deg))

  x2 = (center_x + half_width * cos(angle_deg) - half_height * sin(angle_deg))
  y2 = (center_y + half_width * sin(angle_deg) + half_height * cos(angle_deg))

  x3 = (center_x + half_width * cos(angle_deg) + half_height * sin(angle_deg))
  y3 = (center_y + half_width * sin(angle_deg) - half_height * cos(angle_deg))

  x4 = (center_x - half_width * cos(angle_deg) + half_height * sin(angle_deg))
  y4 = (center_y - half_width * sin(angle_deg) - half_height * cos(angle_deg))

  middle_point1 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
  middle_point2 = (int((x2 + x3) / 2), int((y2 + y3) / 2))
  middle_point3 = (int((x3 + x4) / 2), int((y3 + y4) / 2))
  middle_point4 = (int((x4 + x1) / 2), int((y4 + y1) / 2))

  angle_original = angle
  if width < height:
    angle = 90 - angle
  else:
    angle = -angle
  
  top_point = middle_point1
  if middle_point2[1] < top_point[1]:
    top_point = middle_point2
  if middle_point3[1] < top_point[1]:
    top_point = middle_point3
  if middle_point4[1] < top_point[1]:
    top_point = middle_point4

  return angle, center, top_point, filtered_countours, no_contours_detected

def getDistance(point1, point2):
 
    d = math.sqrt((point2[0] - point1[0])**2 +
                (point2[1] - point1[1])**2)
                #(point2[2] - point1[2])**2)
    return d

def getObj3dCoords(object_center, object_edge, point_cloud):
   
  err, point_cloud_value = point_cloud.get_value(object_edge[0], object_edge[1])
  object_edge_distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                        point_cloud_value[1] * point_cloud_value[1] +
                        point_cloud_value[2] * point_cloud_value[2])
  
  err, point_cloud_value = point_cloud.get_value(object_center[0], object_center[1])
  object_center_distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                        point_cloud_value[1] * point_cloud_value[1] +
                        point_cloud_value[2] * point_cloud_value[2])
  
  return (object_edge[0], object_edge[1], int(object_edge_distance)), (object_center[0], object_center[1], object_center_distance)

def drawContours(img, contours, center, top_point):
    # Draw each contour only for visualisation purposes
  rect = cv2.minAreaRect(contours[0])
  box = cv2.boxPoints(rect)
  box = np.intp(box)

  cv2.drawContours(img,[box],-1,(0,0,255),2)
  
  cv2.circle(img, (center[0], center[1]), 5, (0,255,0), -1)
  #cv22.circle(img, middle_point1, 5, (255, 0, 0), -1)  # Red circle at end point 1
  cv2.circle(img, (top_point[0], top_point[1]), 5, (0, 255, 0), -1)  # Red circle at end point 2

  return img

def checkNewContoursArea(new_contours, old_contours):
   
    if cv2.contourArea(new_contours[0]) != cv2.contourArea[old_contours[0]]:
       return False
    else:
       return True

def getBiggestContours(list_contours): #list_contours[(old_no_contours_detected, old_contours, old_obj_center_point, old_obj_edge_point)]
   
    biggest_contour = None
    center = None
    edge = None
    found_contour = False

    #cycle the list and find the biggest contour
    for item in list_contours:
      if not item[0]:
         found_contour = True
         if biggest_contour is None:
            biggest_contour = item[1]
            center = item[2]
            edge = item[3]
            continue
         new_area = cv2.contourArea(item[1][0])
         old_area = cv2.contourArea(biggest_contour[0])
         div = new_area/old_area
         if  div <=  1.03 * old_area and div >= 0.97 * old_area:
            #print(cv2.contourArea(item[1][0]),cv2.contourArea(biggest_contour[0]))
            biggest_contour = item[1]
            center = item[2]
            edge = item[3]

    return biggest_contour, center, edge, found_contour
         
         
if __name__ == '__main__':

    # Action classifier
    
    action_classifier = SpatioTemporalGCNLearner(device='cpu', dataset_name='custom', method_name=METHOD, num_frames=TARGET_FRAMES,
                                                 in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=9, num_person=1)
    
    model_saved_path = Path(__file__).parent / 'models' / 'sides_200frames' / str(MODEL_TO_TEST) / 'model'
    action_classifier.load(model_saved_path, MODEL_TO_TEST, verbose=True)

    #action_classifier.optimize()

    image_provider = VideoReader()  # loading a video or get the camera id 0

    counter, avg_fps = 0, 0
    poses_list = []
    window = 1
    f_ind = 0
    detector = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity = 1)

    list_contours = []
    missed_hand = False
    hand_center_point = None
    
    for img_rgb, point_cloud in image_provider:
        if f_ind % window == 0:
            start_time = time.perf_counter()

            # angle, obj_center_point, obj_edge_point, contours, no_contours_detected = getOrientation(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            # if not no_contours_detected: 
            #     obj_edge_point, obj_center_point = getObj3dCoords(obj_center_point, obj_edge_point, point_cloud)
            #     list_contours.append((no_contours_detected, contours, obj_center_point, obj_edge_point))
            #     if len(list_contours) > TARGET_FRAMES:
            #         list_contours.pop(0)

            start_time = time.perf_counter()
            img_rgb.flags.writeable = False
            
            results = detector.process(img_rgb)

            pose, hand_center_point, missed_pose, missed_hand = sort_skeleton_data(results, point_cloud)
            counter += 1
            poses_list.append(pose)

            if counter > TARGET_FRAMES: #if more than x00 frames 
                poses_list.pop(0)
                counter = TARGET_FRAMES
            if counter > 0:
                skeleton_seq = pose2numpy(counter, poses_list,3)
                prediction = action_classifier.infer(skeleton_seq)
                category_labels = preds2label(prediction.confidence)
                print(category_labels)
            

            # TODO: alterar o codigo comentado para as novas labels
            # first_key = next(iter(category_labels))
            # first_value = category_labels[first_key]

            # if first_key == 'grab':
            #     contours, obj_center_point, obj_edge_point, found_contour = getBiggestContours(list_contours)
            #     if found_contour:
            #         d_hand_objCenter = getDistance(hand_center_point, obj_center_point)
            #         d_hand_objEdge = getDistance(hand_center_point, obj_edge_point)
            #         # print(obj_center_point)
            #         # print(obj_edge_point)
            #         # print(hand_center_point)
            #         if d_hand_objCenter < d_hand_objEdge:
            #             new_action = 'middle_grab'
            #         else:
            #             new_action = 'edge_grab'
            
            #         #Update predicted labels dictionary
            #         category_labels = {new_action: first_value, **{k: v for k,v in category_labels.items() if k!=new_action and k!= first_key}}
            #contours, obj_center_point, obj_edge_point, found_contour = getBiggestContours(list_contours)


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
            cv2.imshow('Result', annotated_bgr_image)
        f_ind += 1
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        

    #print("Average inference fps: ", avg_fps)
    cv2.destroyAllWindows()
