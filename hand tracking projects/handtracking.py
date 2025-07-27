import cv2
import mediapipe as mp
import time
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

pTime = 0
fingerTips = [4, 8, 12, 16, 20]

# Track previous action to avoid repeated keyDown
previous_action = None

def fingers_up(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[fingerTips[0]].x < hand_landmarks.landmark[fingerTips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tipId in fingerTips[1:]:
        if hand_landmarks.landmark[tipId].y < hand_landmarks.landmark[tipId - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    current_action = "IDLE"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            fingerStates = fingers_up(handLms)

            # GAS: Only index finger up
            if fingerStates == [0, 1, 0, 0, 0]:
                current_action = "GAS"

            # BRAKE: All fingers down
            elif fingerStates == [0, 0, 0, 0, 0]:
                current_action = "BRAKE"

            else:
                current_action = "IDLE"

            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Only send key events when action changes
    if current_action != previous_action:
        # Release all keys first
        pyautogui.keyUp('right')
        pyautogui.keyUp('left')

        if current_action == "GAS":
            pyautogui.keyDown('right')
        elif current_action == "BRAKE":
            pyautogui.keyDown('left')

        previous_action = current_action  # Update state

    # Draw current action on screen
    cv2.putText(img, current_action, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # FPS display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
