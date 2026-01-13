import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

# -------------------------------
# MediaPipe Hands Setup
# -------------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# -------------------------------
# Webcam Setup
# -------------------------------
cap = cv2.VideoCapture(0)

# Virtual Volume Percentage
volPercent = 50

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    h, w, c = img.shape
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS
            )

        # Thumb tip (4) and Index finger tip (8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw gesture line
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Distance between fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # Gesture Logic
        if length > 180:
            pyautogui.press("volumeup")
            volPercent = min(100, volPercent + 2)
            gesture = "Increase Volume"
            feedback = "Good Detection"

        elif length < 60:
            pyautogui.press("volumedown")
            volPercent = max(0, volPercent - 2)
            gesture = "Decrease Volume"
            feedback = "Move Fingers Slightly"

        else:
            gesture = "Stable"
            feedback = "Hold Gesture"

        # -------------------------------
        # UI Display
        # -------------------------------
        volBar = np.interp(volPercent, [0, 100], [400, 150])

        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400),
                      (0, 255, 0), cv2.FILLED)

        cv2.putText(img, f'{volPercent} %',
                    (40, 430),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.putText(img, f'Gesture: {gesture}',
                    (120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.putText(img, f'Feedback: {feedback}',
                    (120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

    cv2.imshow("Milestone 4 - Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
