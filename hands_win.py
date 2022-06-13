# https://www.thekerneltrip.com/python/utils/python-fast-screenshot-locate-on-screen/
# https://automatetheboringstuff.com/chapter18/
# https://www.youtube.com/watch?v=WymCpVUPWQ4
# https://pyautogui.readthedocs.io/en/latest/screenshot.html


import cv2
import mediapipe
import pyautogui
import numpy as np
from PIL import ImageGrab
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(0)

REGION = (-1000, -2000, 500, 500) # (left_x, top_y, right_x, bottom_y)
REGION = (-700, -700, 800, 800) # (left_x, top_y, right_x, bottom_y)

REGION = (0, 0, 400, 400) # (left_x, top_y, right_x, bottom_y)


width, height = pyautogui.size()
print(width, height)
 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
 
    while (True):

        # frame = pyautogui.screenshot(region=(1800,         # left
        #                                   500,         # top
        #                                   width/2,   # width 
        #                                   height/2)) # height
        #frame = ImageGrab.grab(bbox=(500, 500, 600, 700)) # height
        frame = ImageGrab.grab(bbox=REGION) # (left_x, top_y, right_x, bottom_y)

        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
        #ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
 
        cv2.imshow('Test hand', frame)
 
        if cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()
