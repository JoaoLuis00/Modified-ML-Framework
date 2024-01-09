#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from std_msgs.msg import Float32MultiArray
from zed_python.msg import image_depth
import numpy as np


def callback(image_depth_msg):
 
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # Output debugging information to the terminal
  rospy.loginfo("receiving video frame")
   
  # Convert ROS Image message to OpenCV image
  current_frame = br.imgmsg_to_cv2(image_depth_msg.image)
  
  depth_map = image_depth_msg.data
  depth_map = np.reshape(depth_map,(1920,1080))
  #print(depth_map[10,20])
   
  # Display image
  cv2.imshow("camera", current_frame)
   
  cv2.waitKey(1)
      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('video_sub_py', anonymous=True)
  
  # Node is subscribing to the video_frames topic
  #color_sub = message_filters.Subscriber('video_frames', Image)
  rospy.Subscriber('depth_map', image_depth, callback, queue_size=10)
  
  #rospy.Subscriber('depth_map', image_depth, callback)
  #rospy.Subscriber('test', Float32MultiArray, callback_depth_map)
  
  #sync = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)
  #sync.registerCallback(callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()

#!!!CREATE A CUSTOM MESSAGE WITH IMAGE AND DEPTH_MAP TO IGNORE THE NEED TO SYNC
