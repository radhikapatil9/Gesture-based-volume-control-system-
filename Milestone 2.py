import cv2
import mediapipe as mp
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Finger tip IDs with names
finger_tips = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            h, w, _ = img.shape

            # Left or Right Hand
            hand_label = hand_info.classification[0].label
            cv2.putText(
                img,
                f"Hand: {hand_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            lm_list = []

            # Store landmarks
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Show finger names
            for tip_id, finger_name in finger_tips.items():
                x, y = lm_list[tip_id][1], lm_list[tip_id][2]
                cv2.circle(img, (x, y), 8, (255, 0, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    finger_name,
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # Thumb & Index finger distance
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            distance = math.hypot(x2 - x1, y2 - y1)

            cv2.putText(
                img,
                f"Distance: {int(distance)}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

    cv2.imshow("Milestone 2 - Complete Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

