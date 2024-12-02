import cv2
import mediapipe as mp
import pyautogui
import time

# Set the desired mouse speed multiplier and smooth step factor
mouse_speed_multiplier = 1.2  # Reduced speed multiplier
smooth_factor = 10  # Number of steps for smoother movement

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_h, screen_w = pyautogui.size()

safe_margin = 20  # Margin to prevent triggering fail-safe

# Initialize last position for smooth movement
last_x, last_y = screen_w // 2, screen_h // 2

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 225, 0))
            if id == 1:
                # Calculate target position with multiplier
                target_x = min(max(mouse_speed_multiplier * (screen_w / frame_w * x), safe_margin),
                               screen_w - safe_margin)
                target_y = min(max(mouse_speed_multiplier * (screen_h / frame_h * y), safe_margin),
                               screen_h - safe_margin)

                # Smoothly move to the target position
                for step in range(smooth_factor):
                    intermediate_x = last_x + (target_x - last_x) * (step / smooth_factor)
                    intermediate_y = last_y + (target_y - last_y) * (step / smooth_factor)
                    pyautogui.moveTo(intermediate_x, intermediate_y)
                    time.sleep(0.01)  # Adjust sleep duration if needed

                last_x, last_y = target_x, target_y  # Update last position
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 225, 0))
        if (left[0].y - left[1].y) < 0.010:
            pyautogui.click()
            pyautogui.sleep(1)
    cv2.imshow("Eyeball Controlled Cursor", frame)
    cv2.waitKey(1)
