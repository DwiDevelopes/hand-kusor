import cv2
import mediapipe as mp
import pyautogui
import time

# Inisialisasi MediaPipe dan OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inisialisasi video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# Ukuran layar
screen_width, screen_height = pyautogui.size()

# Variabel untuk mengontrol klik
last_click_time = 0
click_delay = 0.1  # Delay dalam detik

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark tangan
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Ambil posisi jari telunjuk
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[8].x * w)  # Jari telunjuk
                y = int(hand_landmarks.landmark[8].y * h)

                # Batasan gerakan kursor agar tidak keluar dari layar
                x = max(0, min(x, screen_width - 1))
                y = max(0, min(y, screen_height - 1))

                # Menggerakkan kursor
                pyautogui.moveTo(x, y)

                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:  
                    hand_status = "Open"
                    
                    if (time.time() - last_click_time) > click_delay:
                        pyautogui.click()
                        last_click_time = time.time()
                else:
                    hand_status = "Closed"

                cursor_radius = 10
                cv2.circle(frame, (x, y), cursor_radius, (0, 255, 255), -1)  

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()