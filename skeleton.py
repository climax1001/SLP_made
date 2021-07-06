import cv2
import os
import numpy as np
import mediapipe as mp
import constants


def load_file_for_skeleton(dirname=constants.dirname, filename=constants.train_dir):
    # 폴더 안의 파일 불러오기
    file_list = [file for file in os.listdir(dirname + filename)]
    image_file = []
    for i in file_list:
        folder_path = dirname + filename + '/' + i
        images = os.listdir(folder_path)
        for image in images:
            image_file.append(image)
            image_file.sort()

        for image in image_file:
            img_file = folder_path + '/' +image
            # img = cv2.imread(img_file)
            # cv2.imshow('IMG', img)
            # cv2.waitKey(50)
            break
    # cv2.destroyAllWindows()
    return img_file

def cv2_hands(IMG_FILES):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For static images:
    IMAGE_FILES = IMG_FILES
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)



img_list = load_file_for_skeleton()
print(img_list)
cv2_hands(img_list)