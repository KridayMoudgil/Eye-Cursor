import cv2
import mediapipe as mp
import pyautogui
import time

# Mouse settings
mouse_speed_multiplier = 3
smooth_factor = 1
safe_margin = 20

# Video input
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_h, screen_w = pyautogui.size()

# Initialize last cursor position
last_x, last_y = screen_w // 2, screen_h // 2

try:
    while True:
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Eye tracking (right eye landmarks 474 to 478)
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 225, 0), -1)

                if id == 1:
                    # Instead of multiplying after scaling, try this:
                    x_offset = (x - frame_w / 2) * mouse_speed_multiplier
                    y_offset = (y - frame_h / 2) * mouse_speed_multiplier

                    target_x = min(max(last_x + x_offset, safe_margin), screen_w - safe_margin)
                    target_y = min(max(last_y + y_offset, safe_margin), screen_h - safe_margin)

                    # Smooth movement
                    for step in range(smooth_factor):
                        intermediate_x = last_x + (target_x - last_x) * (step / smooth_factor)
                        intermediate_y = last_y + (target_y - last_y) * (step / smooth_factor)
                        pyautogui.moveTo(intermediate_x, intermediate_y)
                        time.sleep(0.01)

                    last_x, last_y = target_x, target_y

            # Blink detection (left eye: 145, 159)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 225, 0), -1)

            if (left[0].y - left[1].y) < 0.010:
                pyautogui.click()
                pyautogui.sleep(1)

        cv2.imshow("Eyeball Controlled Cursor", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    cam.release()
    cv2.destroyAllWindows()
