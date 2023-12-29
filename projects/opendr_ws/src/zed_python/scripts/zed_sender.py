#!/usr/bin/env python3
# Basics ROS program to publish real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import pyzed.sl as sl
import os
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from zed_python.msg import image_depth

#image_depth = importlib.import_module('zed-python.msg', 'image_depth', )

#from zed_python.msg import image_depth

def publish_message():
 
  # Node is publishing to the video_frames topic using 
  # the message type Image
  img_pub = rospy.Publisher('video_frames', Image, queue_size=10)
  depth_pub = rospy.Publisher('depth_map', image_depth, queue_size=1)
  test_pub = rospy.Publisher('test', Float32MultiArray, queue_size=10)
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name.
  rospy.init_node('video_pub_py', anonymous=True)
     
  # Go through the loop 10 times per second
  rate = rospy.Rate(60) # 10hz
     
  # Create a VideoCapture object
  # The argument '0' gets the default webcam.
  #cap = cv2.VideoCapture(0)
     
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
  s_name='approach_left_00'
  svo_path = os.path.join('/home/joao/Zed/uncompressed', s_name + '.svo')

  # Create a Camera object
  zed = sl.Camera()

  init_params = sl.InitParameters()
  init_params.svo_real_time_mode = False  # Convert in realtime
  init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

  rt_param = sl.RuntimeParameters()
  rt_param.enable_fill_mode = True

  #Update path to the next svo file
  init_params.set_from_svo_file(f"{svo_path}")

  # Open the camera
  err = zed.open(init_params)
  if err != sl.ERROR_CODE.SUCCESS:
    exit(1)
  
  nb_frames = zed.get_svo_number_of_frames()
  start = time.perf_counter()
  # While ROS is still running.
  while not rospy.is_shutdown():
     
      # Capture frame-by-frame
      # This method returns True/False as well
      # as the video frame.
      
      if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS :
        svo_image = sl.Mat()
        depth_map = sl.Mat()
        zed.retrieve_image(svo_image) #cam.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        #zed.retrieve_image(depth_map, sl.VIEW.DEPTH)
        
        color = cv2.resize(svo_image.get_data(),(int(1920 / 2), int(1080 / 2)))

        #depth_map_msg = create_depth_msg(depth_map)
        
        map = depth_map.get_data()
        map = np.transpose(map,[1,0])
        
        image_depth_msg = image_depth()
        image_depth_msg.image = br.cv2_to_imgmsg(color)
        image_depth_msg.data = map.flatten()

        # Print debugging information to the terminal
        rospy.loginfo('publishing video frame')
        rospy.loginfo('publishing depth map')
        
        svo_position = zed.get_svo_position()
        if svo_position >= (nb_frames - 1):  # End of SVO
          zed.set_svo_position(0)

        img_pub.publish(br.cv2_to_imgmsg(color))
        test_pub.publish(Float32MultiArray(data = map.flatten()))
        depth_pub.publish(image_depth_msg)
        end = time.perf_counter()
        print(end-start)
        start = time.perf_counter()
        #pcl_pub.publish(pcl_msg)
             
      # Sleep just enough to maintain the desired rate
      rate.sleep()

def create_depth_msg(depth_map):
  start = time.perf_counter()
  map = depth_map.get_data()
  map = np.transpose(map,[1,0])
  
  depth_map_msg = Float32MultiArray(data = map.flatten())
  return depth_map_msg

if __name__ == '__main__':
  try:
    publish_message()
  except rospy.ROSInterruptException:
    pass

#[-1.82770483e+03 -1.01410956e+03  2.03229724e+03 -2.06679376e+38]
