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
    print(img_dir_)
IMAGE_FILES = img_dir_
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
            hand_land_data.append(np.nan)
            hand_land_data.append(np.nan)


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
            hand_land_data.append(np.nan)
            hand_land_data.append(np.nan)

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
        pose_land_data.append(np.nan)

    pose_land_data = np.array(pose_land_data).reshape(1,-1)
    hand_land_data = np.array(hand_land_data).reshape(1,-1)

    dfNew = pd.DataFrame(hand_land_data, columns= HLM)
    posedf = pd.DataFrame(pose_land_data, columns=constants.WANNA_POSE)

    skel_data = pd.concat([skel_data, dfNew], ignore_index=True)
    pose_data = pd.concat([pose_data, posedf], ignore_index=True)
    print(skel_data)
    full_data = pd.concat([skel_data, pose_data], axis=1 ,ignore_index=True)
full_data.to_csv("skel_dir/test1.csv", index = False)

    # print('FINGERS')
      # print('INDEX_TIP')
      # print(results.pose_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * image_width,
      #       results.pose_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * image_hight)
      # print('INDEX_DIP')
      # print(results.pose_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x * image_width,
      #       results.pose_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y * image_hight)
    # Draw pose landmarks.
    # annotated_image = image.copy()
    # mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=results.face_landmarks,
    #     connections=mp_holistic.FACE_CONNECTIONS,
    #     landmark_drawing_spec=drawing_spec,
    #     connection_drawing_spec=drawing_spec)
    # mp_drawing.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=results.pose_landmarks,
    #     connections=mp_holistic.POSE_CONNECTIONS,
    #     landmark_drawing_spec=drawing_spec,
    #     connection_drawing_spec=drawing_spec)
    # cv2.imshow('img',annotated_image)
    # cv2.waitKey(10)
# a = torch.Tensor([[3,4,5],[1,2,3]])
# linear_a = nn.Linear(3,64)
# b = torch.Tensor([[2,4,6],[3,6,9]])
# linear_b = nn.Linear(3,64)
# print(a)
# print(a.size(1)) # 3
# # print(linear_(a).shape) # 2x3 * 3x64 = [2,64]
# # print(linear_(a).view(64, 2, -1, 1).transpose(1,2).shape)
# scores_a = linear_a(a).view(64, 2, -1, 1).transpose(1,2)
# scores_b = linear_b(b).view(64, 2, -1, 1).transpose(1,2)
# #
# # scores_a = scores_a / math.sqrt(5)
# # scores = torch.matmul(scores_a, scores_b.transpose(2,3))
# # print(scores.shape) # shape (64,1,2,2)
# # scores = scores.masked_fill
# # test = torch.arange(0, 64).unsqueeze(1)
# # print(test)
#
# encoding_ = torch.zeros(200000, 10)
# position = torch.arange(10, 200000).unsqueeze(1)
# print(encoding_[:,0::2])