import math
import os.path

import pandas as pd
import torch
import torch.nn as nn
import mediapipe as mp
import pandas
import cv2
tmp = ['WRIST','index_finger_dip', 'index_finger_mcp', 'index_finger_dip',
                                  'middel_finger_dip','middel_finger_mcp','middel_finger_pip','middel_finger_tip',
                                  'pinky_dip','pinky_mcp','pinky_pip','pinky_tip',
                                  'ring_dip','ring_mcp','ring_pip','ring_tip',
                                  'thumb_cmc','thumb_ip','thumb_mcp','thumb_tip']
skel_data = pd.DataFrame(columns=['left_HandLandmark.WRIST_x', 'left_HandLandmark.WRIST_y', 'left_HandLandmark.THUMB_CMC_x',
                                  'left_HandLandmark.THUMB_CMC_y', 'left_HandLandmark.THUMB_MCP_x', 'left_HandLandmark.THUMB_MCP_y',
                                  'left_HandLandmark.THUMB_IP_x', 'left_HandLandmark.THUMB_IP_y', 'left_HandLandmark.THUMB_TIP_x', 'left_HandLandmark.THUMB_TIP_y',
                                  'left_HandLandmark.INDEX_FINGER_MCP_x', 'left_HandLandmark.INDEX_FINGER_MCP_y', 'left_HandLandmark.INDEX_FINGER_PIP_x', 'left_HandLandmark.INDEX_FINGER_PIP_y'
    , 'left_HandLandmark.INDEX_FINGER_DIP_x', 'left_HandLandmark.INDEX_FINGER_DIP_y', 'left_HandLandmark.INDEX_FINGER_TIP_x', 'left_HandLandmark.INDEX_FINGER_TIP_y',
                                  'left_HandLandmark.MIDDLE_FINGER_MCP_x', 'left_HandLandmark.MIDDLE_FINGER_MCP_y', 'left_HandLandmark.MIDDLE_FINGER_PIP_x',
                                  'left_HandLandmark.MIDDLE_FINGER_PIP_y', 'left_HandLandmark.MIDDLE_FINGER_DIP_x', 'left_HandLandmark.MIDDLE_FINGER_DIP_y',
                                  'left_HandLandmark.MIDDLE_FINGER_TIP_x', 'left_HandLandmark.MIDDLE_FINGER_TIP_y', 'left_HandLandmark.RING_FINGER_MCP_x',
                                  'left_HandLandmark.RING_FINGER_MCP_y', 'left_HandLandmark.RING_FINGER_PIP_x', 'left_HandLandmark.RING_FINGER_PIP_y',
                                  'left_HandLandmark.RING_FINGER_DIP_x', 'left_HandLandmark.RING_FINGER_DIP_y', 'left_HandLandmark.RING_FINGER_TIP_x', 'left_HandLandmark.RING_FINGER_TIP_y',
                                  'left_HandLandmark.PINKY_MCP_x', 'left_HandLandmark.PINKY_MCP_y', 'left_HandLandmark.PINKY_PIP_x', 'left_HandLandmark.PINKY_PIP_y', 'left_HandLandmark.PINKY_DIP_x', 'left_HandLandmark.PINKY_DIP_y',
                                  'left_HandLandmark.PINKY_TIP_x', 'left_HandLandmark.PINKY_TIP_y'])
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
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
with mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
  for image in IMAGE_FILES:
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    print(image)
    image = cv2.imread(image)
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Print nose coordinates.
    image_hight, image_width, _ = image.shape

    if results.left_hand_landmarks:
        for bodypoint in mp_holistic.HandLandmark:
            print('left')
            print(bodypoint)
            print(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
            skel_data['left_' + str(bodypoint) + '_x'].append(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
            print(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)
            skel_data['left_' + str(bodypoint) + '_y'].append(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)
    if results.right_hand_landmarks:
        for bodypoint in mp_holistic.HandLandmark:
            print('right')
            print(bodypoint)
            print(results.right_hand_landmarks.landmark[bodypoint].x * image_width)
            print(results.right_hand_landmarks.landmark[bodypoint].y * image_hight)

print(skel_data)
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