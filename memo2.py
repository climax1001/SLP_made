import math
import os.path
import time
import pandas as pd
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import pandas
import cv2
import constants

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# For static images:
img_dir = os.listdir('/home/juncislab/PycharmProjects/SLP_made/data')
img_dir.sort()
img_dir_ = []
for dir_list in img_dir:
    a = 'data/' + dir_list
    img_dir_.append(a)
IMAGE_FILES = img_dir_
print(IMAGE_FILES)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

HLM = []
for i in range (0,len(mp_holistic.HandLandmark)):
    HLM.append("LEFT_" + str(list(mp_holistic.HandLandmark)[i]).strip('HandLandmark.') + '_X')
    HLM.append("LEFT_" + str(list(mp_holistic.HandLandmark)[i]).strip('HandLandmark.') + '_Y')
for i in range(0, len(mp_holistic.HandLandmark)):
    HLM.append("RIGHT_" + str(list(mp_holistic.HandLandmark)[i]).strip('HandLandmark.') + '_X')
    HLM.append("RIGHT_" + str(list(mp_holistic.HandLandmark)[i]).strip('HandLandmark.') + '_Y')

skel_data = []
pose_data = []
skel_data = pd.DataFrame(skel_data, columns = HLM)
pose_data = pd.DataFrame(pose_data, columns = constants.WANNA_POSE)
with mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
  for image in IMAGE_FILES:
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    image = cv2.imread(image)
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Print nose coordinates.
    image_hight, image_width, _ = image.shape
    pose_land_data = []
    hand_land_data = []
    if results.left_hand_landmarks:
        for bodypoint in mp_holistic.HandLandmark:
            # left hand 검
            hand_land_data.append(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
            hand_land_data.append(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)
            # print(bodypoint)
            # print(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
            # skel_data['LEFT_' + str(bodypoint).strip('HandLandmark.') + '_X'].append(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
            # print(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)
            # skel_data['LEFT_' + str(bodypoint).strip('HandLandmark.') + '_Y'].append(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)
    else:
        for bodypoint in mp_holistic.HandLandmark:
            hand_land_data.append(0)
            hand_land_data.append(0)


    if results.right_hand_landmarks:
        for bodypoint in mp_holistic.HandLandmark:
            # right hand 검출
            hand_land_data.append(results.right_hand_landmarks.landmark[bodypoint].x * image_width)
            hand_land_data.append(results.right_hand_landmarks.landmark[bodypoint].y * image_hight)
            # print(bodypoint)
            # print(results.right_hand_landmarks.landmark[bodypoint].x * image_width)
            # print(results.right_hand_landmarks.landmark[bodypoint].y * image_hight)
    else:
        for bodypoint in mp_holistic.HandLandmark:
            hand_land_data.append(0)
            hand_land_data.append(0)

    if results.pose_landmarks:
        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width)
        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_hight)

        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width)
        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_hight)

        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_hight)

        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
        pose_land_data.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
        # hand_land_data.append(results.pose_landmarks.landmark[])
    else:
        for i in range(8):
            pose_land_data.append(0)

    pose_land_data = np.array(pose_land_data).reshape(-1,8)
    print(pose_land_data)
    hand_land_data = np.array(hand_land_data).reshape(-1,84)
    print(hand_land_data)
    dfNew = pd.DataFrame(hand_land_data, columns= HLM)
    posedf = pd.DataFrame(pose_land_data, columns=constants.WANNA_POSE)

    skel_data = pd.concat([skel_data, dfNew], ignore_index=True)
    pose_data = pd.concat([pose_data, posedf], ignore_index=True)
    print(skel_data)
    full_data = pd.concat([skel_data, pose_data], axis=1 ,ignore_index=True)
full_data.to_csv("skel_dir/test1.csv", index = False)
