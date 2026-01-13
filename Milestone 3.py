import cv2
import mediapipe as mp
import math
import pyautogui
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Delay to avoid too fast volume change
last_time = time.time()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape

            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Thumb tip (4) and Index tip (8)
            x1, y1 = lm_list[4]
            x2, y2 = lm_list[8]

            # Draw
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Distance
            distance = math.hypot(x2 - x1, y2 - y1)

            # Control volume every 0.2 seconds
            if time.time() - last_time > 0.2:
                if distance > 150:
                    pyautogui.press("volumeup")
                elif distance < 50:
                    pyautogui.press("volumedown")
                last_time = time.time()

            # Display distance
            cv2.putText(img, f'Distance: {int(distance)}',
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Volume Control - PyAutoGUI", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
