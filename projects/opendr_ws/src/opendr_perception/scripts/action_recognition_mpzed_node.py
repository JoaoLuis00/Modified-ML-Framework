import argparse
import rospy
import torch
import numpy as np
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesis
from opendr_bridge.msg import OpenDRPose2D
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from zed_python.msg import image_depth
from opendr.perception.skeleton_based_action_recognition import SpatioTemporalGCNLearner
from opendr.engine.data import Image
import mediapipe as mp
from pathlib import Path
from cv_bridge import CvBridge
import cv2
import pandas as pd
from typing import Dict
from opendr.engine.target import MPPose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

TARGET_FRAMES = 250
NUM_KEYPOINTS = 46
ORIGIN = (0,0,0)
WRIST = mp.solutions.holistic.HandLandmark.WRIST
LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW

poses_list = []
counter = 0

DATA_TYPE = 'final_atualizado_fullsize'
MODEL_TO_TEST = 'tagcn_25epochs_0.01lr_75subframes_dropafterepoch3040_batch30'

class SkeletonActionRecognitionNode:

    def __init__(self):
        """
        Creates a ROS Node for skeleton-based action recognition
        :param input_rgb_image_topic: Topic from which we are reading the input image
        :type input_rgb_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated image (if None, we are not publishing
        annotated image)
        :type output_rgb_image_topic: str
        :param pose_annotations_topic: Topic to which we are publishing the annotations (if None, we are not publishing
        annotated pose annotations)
        :type pose_annotations_topic:  str
        :param output_category_topic: Topic to which we are publishing the recognized action category info
        (if None, we are not publishing the info)
        :type output_category_topic: str
        :param output_category_description_topic: Topic to which we are publishing the description of the recognized
        action (if None, we are not publishing the description)
        :type output_category_description_topic:  str
        :param device: device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param model:  model to use for skeleton-based action recognition.
         (Options: 'stgcn', 'pstgcn')
        :type model: str
        """

        # Set up ROS topics and bridge
        self.input_data_topic = "depth_map"
        self.bridge = CvBridge()

        self.image_publisher = rospy.Publisher("/action_recognition/annotated_image", ROS_Image, queue_size=1)

        self.hypothesis_publisher = rospy.Publisher("/action_recognition/category", ObjectHypothesis, queue_size=1)

        self.hypothesis_publisher = None

        self.string_publisher = rospy.Publisher("/action_recognition/category_description", String, queue_size=1)

        # Initialize the pose estimation
        self.pose_estimator = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity = 1)

        action_classifier = SpatioTemporalGCNLearner(device='cpu', dataset_name='custom', method_name='tagcn', num_frames=TARGET_FRAMES,
                                                 in_channels=3,num_point=NUM_KEYPOINTS, graph_type='custom', num_class=6, num_person=1)
        
        model_saved_path = Path(__file__).parent / 'models' / str(DATA_TYPE) / str(MODEL_TO_TEST) / 'model'
        action_classifier.load(model_saved_path, MODEL_TO_TEST, verbose=True)
        

    def listen(self):
        """
        Start the node and begin processing input data
        """
        rospy.init_node('opendr_skeleton_action_recognition_node', anonymous=True)
        rospy.Subscriber(self.input_data_topic, image_depth, self.callback, queue_size=1, buff_size=10000000)
        rospy.loginfo("Skeleton-based action recognition node started.")
        rospy.spin()

    def callback(self, data):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param data: input message
        :type data: sensor_msgs.msg.Image
        """

        # Convert sensor_msgs.msg.Image into OpenDR Image
        image = self.bridge.imgmsg_to_cv2(data.image)

        # Run pose estimation
        results = self.pose_estimator.process(image)
        
        #Unflatten the array
        depth_map = np.reshape(data.data,(1920,1080))
        
        pose, left_hand_center_point, right_hand_center_point, missed_pose, missed_hand_left, missed_hand_right = sort_skeleton_data(results, depth_map)

        annotated_img = draw_skeletons(image, results)
        
        if self.image_publisher is not None:
            message = self.bridge.to_ros_image(Image(annotated_img))
            self.image_publisher.publish(message)

        
        if not missed_pose and not missed_hand_left and not missed_hand_right:
            counter += 1
            poses_list.append(pose)
            skeleton_seq = pose2numpy(counter, poses_list)
            if counter > TARGET_FRAMES: #if more than x frames 
                poses_list.pop(0)
                counter = TARGET_FRAMES
            if counter > 0:
                skeleton_seq = pose2numpy(counter, poses_list,3)
                category = self.action_classifier.infer(skeleton_seq)
            
            category.confidence = float(category.confidence.max())

            if self.hypothesis_publisher is not None:
                self.hypothesis_publisher.publish(to_ros_category(category))

            if self.string_publisher is not None:
                self.string_publisher.publish(to_ros_category_description(category))

def sort_skeleton_data(results, depth_map):
    pose_keypoints = np.ones((NUM_KEYPOINTS, 3), dtype=np.int32) * -1
    hand_keypoints_list_left = []
    hand_keypoints_list_right = []
    missed_hand_left = missed_hand_right = missed_pose = False
    mean_x_left = mean_y_left = mean_z_left = -1
    mean_x_right = mean_y_right = mean_z_right = -1

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
    
        try:  # change the z coordinate to get from the generated depth information
            # Left Shoulder
            pose_keypoints[0, 0] = c_x = round(pose_landmarks[LEFT_SHOULDER].x * 1920)
            pose_keypoints[0, 1] = c_y = round(pose_landmarks[LEFT_SHOULDER].y * 1080)
            err, depth_val = depth_map[c_x,c_y]
            pose_keypoints[0, 2] = round(depth_val)
        except (IndexError, AttributeError) as e:
            missed_pose = True

        try:
            # Right Shoulder
            pose_keypoints[1, 0] = c_x = round(pose_landmarks[RIGHT_SHOULDER].x * 1920)
            pose_keypoints[1, 1] = c_y = round(pose_landmarks[RIGHT_SHOULDER].y * 1080)
            err, depth_val = depth_map[c_x,c_y]
        except (IndexError, AttributeError) as e:
            missed_pose = True
            
        try:
            # Left Elbow
            pose_keypoints[2, 0] = c_x = round(pose_landmarks[LEFT_ELBOW].x * 1920)
            pose_keypoints[2, 1] = c_y = round(pose_landmarks[LEFT_ELBOW].y * 1080)
            err, depth_val = depth_map[c_x,c_y]
            pose_keypoints[2, 2] = round(depth_val)
        except (IndexError, AttributeError) as e:
            missed_pose = True
            
        try:
            # Right Elbow
            pose_keypoints[3, 0] = c_x = round(pose_landmarks[RIGHT_ELBOW].x * 1920)
            pose_keypoints[3, 1] = c_y = round(pose_landmarks[RIGHT_ELBOW].y * 1080)
            err, depth_val = depth_map[c_x,c_y]
            pose_keypoints[3, 2] = round(depth_val)
        except (IndexError, AttributeError) as e:
            missed_pose = True
        
    if not results.left_hand_landmarks:
        missed_hand_left = True
    else:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        #Left Wrist
        pose_keypoints[4, 0] = c_x = round(left_hand_landmarks[WRIST].x * 1920)
        pose_keypoints[4, 1] = c_y = round(left_hand_landmarks[WRIST].y * 1080)
        err, depth_val = depth_map[c_x,c_y]
        d_lwrist = depth_val
        pose_keypoints[4, 2] = round(d_lwrist)
        
        for idx, landmark in enumerate(left_hand_landmarks[1:]):
            x, y, z = landmark.x * 1920, landmark.y * 1080, d_lwrist + d_lwrist*landmark.z
            pose_keypoints[idx+5, 0] = round(x)
            pose_keypoints[idx+5, 1] = round(y)
            pose_keypoints[idx+5, 2] = round(z) 

    if not results.right_hand_landmarks:
        missed_hand_right = True
    else:
        right_hand_landmarks =  results.right_hand_landmarks.landmark
        #Right Wrist
        pose_keypoints[25, 0] = c_x = round(right_hand_landmarks[WRIST].x * 1920)
        pose_keypoints[25, 1] = c_y = round(right_hand_landmarks[WRIST].y * 1080)
        err, depth_val = depth_map[c_x,c_y]
        d_rwrist = depth_val
        pose_keypoints[25, 2] = round(d_rwrist)
        
        for idx, landmark in enumerate(right_hand_landmarks[1:]):
            x, y, z = landmark.x * 1920, landmark.y * 1080, d_rwrist + d_rwrist*landmark.z
            pose_keypoints[idx+26, 0] = round(x)
            pose_keypoints[idx+26, 1] = round(y)
            pose_keypoints[idx+26, 2] = round(z)

    pose = MPPose(pose_keypoints, -1)

    if not missed_hand_left:
        mean_x_left = sum(coord[0] for coord in hand_keypoints_list_left) / len(hand_keypoints_list_left)
        mean_y_left = sum(coord[1] for coord in hand_keypoints_list_left) / len(hand_keypoints_list_left)
        mean_z_left = sum(coord[2] for coord in hand_keypoints_list_left) / len(hand_keypoints_list_left)
        left_hand_center_point = (mean_x_left, mean_y_left, mean_z_left)
        
    if not missed_hand_right:
        mean_x_right = sum(coord[0] for coord in hand_keypoints_list_right) / len(hand_keypoints_list_right)
        mean_y_right = sum(coord[1] for coord in hand_keypoints_list_right) / len(hand_keypoints_list_right)
        mean_z_right = sum(coord[2] for coord in hand_keypoints_list_right) / len(hand_keypoints_list_right)
        right_hand_center_point = (mean_x_right, mean_y_right, mean_z_right)

    return pose, left_hand_center_point,right_hand_center_point, missed_pose, missed_hand_left, missed_hand_right

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

# def preds2label(confidence):
#     k = 3
#     class_scores, class_inds = torch.topk(confidence, k=k)
#     labels = {ACTION_CLASSES[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
#     return labels

def draw_preds(frame, preds: Dict):
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(frame, f"{prob:04.3f} {cls}",
                    (10, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2,)
        
def to_ros_category_description(category):
        """
        Converts an OpenDR category into a string msg that can carry the Category.description.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS message with the category.description
        :rtype: std_msgs.msg.String
        """
        result = String()
        result.data = category.description
        return result
    
def to_ros_category(category):
        """
        Converts an OpenDR category into a ObjectHypothesis msg that can carry the Category.data and Category.confidence.
        :param category: OpenDR category to be converted
        :type category: engine.target.Category
        :return: ROS message with the category.data and category.confidence
        :rtype: vision_msgs.msg.ObjectHypothesis
        """
        result = ObjectHypothesis()
        result.id = category.data
        result.score = category.confidence
        return result
    
if __name__ == '__main__':
    
    try:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"
        
    skeleton_action_recognition_node = \
        SkeletonActionRecognitionNode()
    skeleton_action_recognition_node.listen()
