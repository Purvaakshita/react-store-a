import cv2
import numpy as np
import mediapipe as mp
import webbrowser
import tkinter as tk
from tkinter import simpledialog
from collections import deque

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  
colorIndex = 0  
points = [deque(maxlen=1024)]
button_positions = [(10, 1, 80, 50), (90, 1, 160, 50), (170, 1, 240, 50), (250, 1, 320, 50), (330, 1, 400, 50), (410, 1, 480, 50), (490, 1, 560, 50)]
button_texts = ["Blue", "Green", "Red", "Yellow", "Clear", "Web", "Table"]

shape_positions = [(570, 60, 630, 110), (570, 120, 630, 170), (570, 180, 630, 230), (570, 240, 630, 290)]
shape_texts = ["Square", "Circle", "Rect", "Tri"]
selected_shape = None
shape_position = None  
size = 50  
size_adjusting = False  

def open_website():
    root = tk.Tk()
    root.withdraw()
    url = simpledialog.askstring("Input", "Enter website URL:")
    if url:
        webbrowser.open(url)

def draw_table(rows, cols):
    cell_width = paintWindow.shape[1] // (cols + 1)
    cell_height = paintWindow.shape[0] // (rows + 1)
    for i in range(1, rows + 1):
        y = i * cell_height
        cv2.line(paintWindow, (0, y), (paintWindow.shape[1], y), (0, 0, 0), 2)
    for j in range(1, cols + 1):
        x = j * cell_width
        cv2.line(paintWindow, (x, 0), (x, paintWindow.shape[0]), (0, 0, 0), 2)

def create_table():
    root = tk.Tk()
    root.withdraw()
    rows = simpledialog.askinteger("Input", "Enter number of rows:")
    cols = simpledialog.askinteger("Input", "Enter number of columns:")
    if rows and cols:
        draw_table(rows, cols)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

paintWindow = np.ones((471, 640, 3), dtype=np.uint8) * 255
cv2.namedWindow("Paint")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    for i, (x1, y1, x2, y2) in enumerate(button_positions):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i] if i < 4 else (50, 50, 50), -1)
        text_size = cv2.getTextSize(button_texts[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2 + 5
        cv2.putText(frame, button_texts[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (x1, y1, x2, y2) in enumerate(shape_positions):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), -1)
        text_size = cv2.getTextSize(shape_texts[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2 + 5
        cv2.putText(frame, shape_texts[i], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in handLms.landmark]
            
            if len(lmList) > 8:
                index_finger = lmList[8]
                thumb = lmList[4]

                for i, (x1, y1, x2, y2) in enumerate(button_positions):
                    if x1 < index_finger[0] < x2 and y1 < index_finger[1] < y2:
                        if i == 4:
                            points = [deque(maxlen=1024)]
                            paintWindow[:] = 255
                            selected_shape = None
                            shape_position = None
                        elif i == 5:
                            open_website()
                        elif i == 6:
                            create_table()
                        else:
                            colorIndex = i
                        break

                for i, (x1, y1, x2, y2) in enumerate(shape_positions):
                    if x1 < index_finger[0] < x2 and y1 < index_finger[1] < y2:
                        selected_shape = shape_texts[i]
                        shape_position = (frame.shape[1] // 2, frame.shape[0] // 2)  
                        size_adjusting = True  
                        break

                distance = np.linalg.norm(np.array(index_finger) - np.array(thumb))
                if size_adjusting:
                    if distance < 20:
                        size = max(20, size - 2)
                    elif distance > 60:
                        size = min(200, size + 2)
                    else:
                        size_adjusting = False  

                if selected_shape and shape_position:
                    x, y = shape_position
                    if selected_shape == "Square":
                        cv2.rectangle(paintWindow, (x - size, y - size), (x + size, y + size), colors[colorIndex], 2)
                    elif selected_shape == "Circle":
                        cv2.circle(paintWindow, (x, y), size, colors[colorIndex], 2)
                    elif selected_shape == "Rect":
                        cv2.rectangle(paintWindow, (x - size * 2, y - size), (x + size * 2, y + size), colors[colorIndex], 2)
                    elif selected_shape == "Tri":
                        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
                        cv2.polylines(paintWindow, [pts], isClosed=True, color=colors[colorIndex], thickness=2)

                if selected_shape is None:
                    if distance > 40:  
                        points[-1].appendleft(index_finger)
                    else:
                        points.append(deque(maxlen=1024))

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    else:
        points.append(deque(maxlen=1024))
    
    for stroke in points:
        for i in range(1, len(stroke)):
            if stroke[i - 1] is None or stroke[i] is None:
                continue
            cv2.line(frame, stroke[i - 1], stroke[i], colors[colorIndex], 8, cv2.LINE_AA)
            cv2.line(paintWindow, stroke[i - 1], stroke[i], colors[colorIndex], 8, cv2.LINE_AA)
    
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
