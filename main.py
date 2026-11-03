import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Automatically download the required MediaPipe model file if not present
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand_landmarker.task model (one-time setup)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# Initialize Modern MediaPipe Hand Landmarker with optimized stability
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,  # Lowered for better tracking grip
    min_tracking_confidence=0.6       # Lowered to prevent tracking dropouts
)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Canvas to store permanent drawings
paintWindow = np.zeros((720, 1280, 3), dtype=np.uint8)

# Color Palette Definition (BGR format)
COLORS = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Eraser": (0, 0, 0)
}
color_names = list(COLORS.keys())
active_color_name = "Blue"
brush_color = COLORS[active_color_name]

brush_thickness = 10
eraser_thickness = 50
prev_x, prev_y = 0, 0

# Smoothing variables for fluid cursor tracking
smooth_factor = 0.6  # Slightly higher responsiveness
curr_x, curr_y = 0, 0
target_x, target_y = 0, 0

# Buffer to tolerate temporary tracking loss (prevents line breakage)
lost_frame_count = 0
MAX_LOST_FRAMES = 5  

# Button selection hover timer
hover_start_time = 0
selected_box_index = -1
HOVER_DELAY = 0.6

print("--- Stabilized Air Draw App Started ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip frame horizontally for natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert frame to MediaPipe Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    drawing_mode = False
    in_header_zone = False
    
    # --- Draw UI Color Palette Boxes at Top ---
    box_width = 120
    box_height = 60
    start_x = (w - (len(COLORS) * 140)) // 2
    
    for i, (name, col) in enumerate(COLORS.items()):
        bx = start_x + (i * 140)
        by = 20
        box_color = (255, 255, 255) if name != "Eraser" else (100, 100, 100)
        cv2.rectangle(frame, (bx, by), (bx + box_width, by + box_height), box_color, 2)
        
        if name == "Eraser":
            cv2.rectangle(frame, (bx + 5, by + 5), (bx + box_width - 5, by + box_height - 5), (50, 50, 50), -1)
            cv2.putText(frame, "ERASE", (bx + 22, by + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (bx + 5, by + 5), (bx + box_width - 5, by + box_height - 5), col, -1)
            
        cv2.putText(frame, str(i+1), (bx + 10, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if active_color_name == name:
            cv2.rectangle(frame, (bx - 4, by - 4), (bx + box_width + 4, by + box_height + 4), (0, 255, 0), 3)

    if result.hand_landmarks:
        lost_frame_count = 0  # Reset lost counter when hand is detected
        for hand_landmarks in result.hand_landmarks:
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            
            tx, ty = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            
            # Smooth out coordinates
            target_x, target_y = tx, ty
            curr_x = int(curr_x + (target_x - curr_x) * smooth_factor)
            curr_y = int(curr_y + (target_y - curr_y) * smooth_factor)
            
            if curr_y < 100:
                in_header_zone = True
                prev_x, prev_y = 0, 0
                
                hovered_index = -1
                for i in range(len(COLORS)):
                    bx = start_x + (i * 140)
                    if bx <= curr_x <= bx + box_width and 20 <= curr_y <= 20 + box_height:
                        hovered_index = i
                        break
                
                if hovered_index != -1:
                    if selected_box_index == hovered_index:
                        if cv2.getTickCount() / cv2.getTickFrequency() - hover_start_time > HOVER_DELAY:
                            active_color_name = color_names[hovered_index]
                            brush_color = COLORS[active_color_name]
                            print(f"Active Color Changed to: {active_color_name}")
                            selected_box_index = -1
                    else:
                        selected_box_index = hovered_index
                        hover_start_time = cv2.getTickCount() / cv2.getTickFrequency()
                else:
                    selected_box_index = -1
            else:
                selected_box_index = -1
                drawing_mode = ty < (my - 30)
                
                if drawing_mode:
                    current_thickness = eraser_thickness if active_color_name == "Eraser" else brush_thickness
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = curr_x, curr_y
                    
                    if active_color_name == "Eraser":
                        cv2.line(paintWindow, (prev_x, prev_y), (curr_x, curr_y), (0, 0, 0), current_thickness)
                    else:
                        cv2.line(paintWindow, (prev_x, prev_y), (curr_x, curr_y), brush_color, current_thickness)
                    
                    prev_x, prev_y = curr_x, curr_y
                else:
                    prev_x, prev_y = 0, 0
    else:
        # Grace period for brief tracking drops to avoid breaking the line instantly
        lost_frame_count += 1
        if lost_frame_count > MAX_LOST_FRAMES:
            prev_x, prev_y = 0, 0

    # Merge drawing canvas with the live camera feed
    gray_canvas = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, paintWindow)

    # --- Cursor Visualization ---
    if result.hand_landmarks:
        if in_header_zone:
            cv2.circle(frame, (curr_x, curr_y), 12, (0, 255, 255), 2)
        elif drawing_mode:
            curr_thick = eraser_thickness if active_color_name == "Eraser" else brush_thickness
            cv2.circle(frame, (curr_x, curr_y), curr_thick, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (curr_x, curr_y), 8, (255, 255, 255), 2)

    # Bottom HUD Status Bar
    cv2.putText(frame, f"Active: {active_color_name} [1-5: Color] | C: Clear | S: Save | Q: Quit", (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    # Display Output Window
    cv2.imshow("Stabilized Air Draw", frame)

    # Keyboard controls & Instant Color Switching
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        paintWindow = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("Canvas Cleared!")
    elif key == ord('s'):
        cv2.imwrite("colored_air_drawing.png", paintWindow)
        print("Drawing saved successfully!")
    elif key == ord('1'):
        active_color_name = "Blue"
        brush_color = COLORS[active_color_name]
    elif key == ord('2'):
        active_color_name = "Green"
        brush_color = COLORS[active_color_name]
    elif key == ord('3'):
        active_color_name = "Red"
        brush_color = COLORS[active_color_name]
    elif key == ord('4'):
        active_color_name = "Yellow"
        brush_color = COLORS[active_color_name]
    elif key == ord('5'):
        active_color_name = "Eraser"

cap.release()
detector.close()
cv2.destroyAllWindows()