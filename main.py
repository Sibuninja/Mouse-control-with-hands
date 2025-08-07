import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture video
cap = cv2.VideoCapture(0)

# Smooth cursor movement
prev_x, prev_y = 0, 0
smoothening = 5

click_state = {'left': False, 'right': False, 'dragging': False}
scroll_threshold = 40  # Distance between fingers for scroll detection

def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    lm_list = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return lm_list

def fingers_up(lm_list):
    tips = [8, 12, 16, 20]
    up = []
    if not lm_list:
        return [0, 0, 0, 0, 0]
    # Thumb
    up.append(1 if lm_list[4][0] > lm_list[3][0] else 0)
    # Fingers
    for tip in tips:
        up.append(1 if lm_list[tip][1] < lm_list[tip - 2][1] else 0)
    return up

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lm_list = get_landmarks(img)

    if len(lm_list) != 0:
        index_finger = lm_list[8]
        middle_finger = lm_list[12]
        ring_finger = lm_list[16]
        pinky = lm_list[20]

        # Map to screen size
        x = np.interp(index_finger[0], (100, 540), (0, screen_width))
        y = np.interp(index_finger[1], (100, 380), (0, screen_height))

        curr_x = prev_x + (x - prev_x) / smoothening
        curr_y = prev_y + (y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Detect finger states
        fingers = fingers_up(lm_list)

        # Left Click: Index down only
        if fingers[1] == 0 and fingers[2] == 1:
            if not click_state['left']:
                click_state['left'] = True
                pyautogui.click()
        else:
            click_state['left'] = False

        # Right Click: Middle down only
        if fingers[2] == 0 and fingers[1] == 1:
            if not click_state['right']:
                click_state['right'] = True
                pyautogui.click(button='right')
        else:
            click_state['right'] = False

        # Drag and drop: Index and thumb touching (distance < threshold)
        if fingers[1] == 1 and fingers[0] == 1:
            if distance(lm_list[4], lm_list[8]) < 40:
                if not click_state['dragging']:
                    click_state['dragging'] = True
                    pyautogui.mouseDown()
            else:
                if click_state['dragging']:
                    click_state['dragging'] = False
                    pyautogui.mouseUp()

        # Scroll up
        if distance(middle_finger, ring_finger) > scroll_threshold:
            pyautogui.scroll(20)
        # Scroll down
        elif distance(middle_finger, pinky) > scroll_threshold:
            pyautogui.scroll(-20)

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
