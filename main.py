import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def find_angle(c1, c2, c3):
    if (c2[0] - c1[0]) != 0 and (c3[0] - c2[0]) != 0:
        slopes = (c2[1] - c1[1]) / (c2[0] - c1[0]), (c3[1] - c2[1]) / (c3[0] - c2[0])
    else:
        slopes = 0, 0
        print("Division by zero")

    tan_theta = 0
    if 1 + (slopes[0] * slopes[1]) != 0:
        tan_theta = (slopes[1] - slopes[0]) / (1 + (slopes[0] * slopes[1]))

    angle = math.degrees(math.atan(tan_theta))
    while angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return angle


while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # working with each hand
            cords = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # NUMBER LABELS
                # image = cv2.putText(image, str(id), (cx + 50, cy - 50),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 35, 50), 2, cv2.LINE_AA)

                # pinkie = 0, 17, 18, 19, 20
                cords.append((cx, cy))

                # PINKIE MARKER
                #if id == 20:
                #     cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            angles = find_angle(cords[0], cords[17], cords[18]), find_angle(cords[17], cords[18], cords[19]), find_angle(cords[18], cords[19], cords[20])

            # DRAW ANGLES
            image = cv2.putText(image, str(angles), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 35, 50), 2, cv2.LINE_AA)

            strength = int((angles[0] + angles[1] + angles[0]) / 3)
            image = cv2.putText(image, "GRIP STRENGTH: " + str(strength), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 35, 255), 4, cv2.LINE_AA)

            # DRAW CONNECTIONS
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Output", image)
    cv2.waitKey(1)

