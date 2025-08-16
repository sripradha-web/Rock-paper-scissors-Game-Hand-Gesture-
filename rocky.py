import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to detect gesture
def detect_gesture(hand_landmarks):
    # Get y-coordinates for finger tips and their lower joints
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    joints = [6, 10, 14, 18]  # Corresponding lower joints

    fingers_up = []
    for tip, joint in zip(tips, joints):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[joint].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    thumb_tip = hand_landmarks.landmark[4]
    thumb_joint = hand_landmarks.landmark[3]
    thumb_open = thumb_tip.x > thumb_joint.x

    # Classify gesture
    if fingers_up == [0, 0, 0, 0]:
        return "Rock"
    elif fingers_up == [1, 1, 1, 1]:
        return "Paper"
    elif fingers_up == [1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

# Decide winner
def get_winner(player_move, computer_move):
    if player_move == computer_move:
        return "It's a Draw"
    elif (player_move == "Rock" and computer_move == "Scissors") or \
         (player_move == "Scissors" and computer_move == "Paper") or \
         (player_move == "Paper" and computer_move == "Rock"):
        return "You Win"
    else:
        return "Computer Wins"

cap = cv2.VideoCapture(0)
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    player_move = "Waiting..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            player_move = detect_gesture(hand_landmarks)

    # Computer randomly picks every 3 seconds
    if time.time() - last_time > 3 and player_move != "Waiting..." and player_move != "Unknown":
        computer_move = random.choice(["Rock", "Paper", "Scissors"])
        result = get_winner(player_move, computer_move)

        print(f"You: {player_move} | Computer: {computer_move} --> {result}")
        last_time = time.time()

    cv2.putText(frame, f"Your Move: {player_move}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

