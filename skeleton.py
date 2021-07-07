import cv2
import os
import numpy as np
import mediapipe as mp
import constants
import pandas as pd

def load_file_for_skeleton(dirname=constants.dirname, filename=constants.train_dir):
    # 폴더 안의 파일 불러오기
    file_list = [file for file in os.listdir(dirname + filename)]
    path_files = []
    name_files = []

    for i in file_list:
        folder_path = dirname + filename + '/' + i
        path_files.append(folder_path)
        file_name = folder_path.split('/')[-1]
        name_files.append(file_name)

    return path_files, name_files

def get_skeleton_csv(img_file_path):

    img_file_path.sort()

    HLM = []
    POS = constants.WANNA_POSE

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
        for image in img_file_path:
            image = cv2.imread(image)
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_hight, image_width, _ = image.shape
            pose_land_data = []
            hand_land_data = []
            if results.left_hand_landmarks:
                for bodypoint in mp_holistic.HandLandmark:
                    hand_land_data.append(results.left_hand_landmarks.landmark[bodypoint].x * image_width)
                    hand_land_data.append(results.left_hand_landmarks.landmark[bodypoint].y * image_hight)

            else:
                    hand_land_data.append(np.nan)
                    hand_land_data.append(np.nan)

            if results.right_hand_landmarks:
                for bodypoint in mp_holistic.HandLandmark:
                    hand_land_data.append(results.right_hand_landmarks.landmark[bodypoint].x * image_width)
                    hand_land_data.append(results.right_hand_landmarks.landmark[bodypoint].y * image_hight)

            else:
                    hand_land_data.append(np.nan)
                    hand_land_data.append(np.nan)

            if results.pose_landmarks:
                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width)
                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_hight)

                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width)
                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_hight)

                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_hight)

                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
                pose_land_data.append(
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
            else:
                pose_land_data.append(np.nan)

            pose_land_data = np.array(pose_land_data).reshape(1, -1)
            hand_land_data = np.array(hand_land_data).reshape(1, -1)

            dfNew = pd.DataFrame(hand_land_data, columns=HLM)
            posedf = pd.DataFrame(pose_land_data, columns=constants.WANNA_POSE)

            skel_data = pd.concat([skel_data, dfNew], ignore_index=True)
            pose_data = pd.concat([pose_data, posedf], ignore_index=True)
            full_data = pd.concat([skel_data, pose_data], axis=1, ignore_index=True)

    return full_data.to_csv("skel_dir/test1.csv", index=False)

def get_files(paths):
    for path in paths:
        full_path = []
        img_list = os.listdir(path)
        img_list.sort()
        for i in img_list:
            fp = path + '/' + str(i)
            full_path.append(fp)
        print(full_path)
        get_skeleton_csv(full_path)

paths, name = load_file_for_skeleton()
get_files(paths)