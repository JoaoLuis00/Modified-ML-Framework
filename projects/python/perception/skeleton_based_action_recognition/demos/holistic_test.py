import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

video_path = '/hoe/joao/videos/approach_50_rgb.avi'
cv2.imread('/home/joao/Zed/frames/approach_10/frame_0.png')
# For static images:
cap = cv2.VideoCapture(video_path)
with mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.4, model_complexity = 2) as holistic:
    start_time = time.time()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            cap.release()
            end_time = time.time()
            print(f'Total time: {end_time-start_time}')
            cv2.destroyAllWindows
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

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
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    