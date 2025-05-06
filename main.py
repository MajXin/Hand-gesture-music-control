import cv2
import mediapipe as mp
import pygame
import math
import time
import numpy as np

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("rock.mp3")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# State
is_playing = True
last_toggle_time = time.time()
cooldown = 1.5  # seconds

pygame.mixer.music.play(-1)  # Loop the music
pygame.mixer.music.set_volume(0.5)  # Initial volume

def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    folded = sum(1 for tip, pip in zip(finger_tips, finger_pips)
                 if landmarks[tip].y > landmarks[pip].y)
    thumb_folded = abs(landmarks[4].x - landmarks[3].x) < 0.05
    return folded >= 3 and thumb_folded

def is_open_fist(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return distance > 0.15

def calculate_volume(thumb_tip, index_tip):
    distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    # Scale distance: min=0.02 (close), max=0.25 (far), clamp between 0 and 1
    volume = np.interp(distance, [0.02, 0.25], [0.0, 1.0])
    volume = np.clip(volume, 0.0, 1.0)
    return volume

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Volume control
            volume = calculate_volume(thumb_tip, index_tip)
            pygame.mixer.music.set_volume(volume)
            vol_percent = int(volume * 100)
            cv2.putText(frame, f'Volume: {vol_percent}%', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Play/pause toggle
            current_time = time.time()
            if current_time - last_toggle_time > cooldown:
                if is_playing and is_fist(landmarks):
                    pygame.mixer.music.pause()
                    is_playing = False
                    last_toggle_time = current_time
                    print("Paused")
                    cv2.putText(frame, "Fist - Paused", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                elif not is_playing and is_open_fist(landmarks):
                    pygame.mixer.music.unpause()
                    is_playing = True
                    last_toggle_time = current_time
                    print("Resumed")
                    cv2.putText(frame, "Open - Resumed", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gesture Music Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pygame.mixer.music.stop()
        break

cap.release()
cv2.destroyAllWindows()
