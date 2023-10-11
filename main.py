import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import math
import time
from speed_estimate import *
import os

path = r"C:\Users\doant\Downloads\traffic_-_27260 (Original).mp4"
path_yolo = r"C:\Users\doant\Downloads\yolov8n.pt"

cap = cv2.VideoCapture(path)
ret, frame_plt = cap.read()
frame_plt = cv2.resize(frame_plt, (800, 500))
frame_plt_shape = frame_plt.copy()

width_line1_x1 = 263
width_line2_x2 = 547
height_line1_y1 = 144
height_line2_y2 = 136

roi_y1 = 100
roi_y2 = int(frame_plt.shape[0])
roi_x1 = 0
roi_x2 = int(frame_plt.shape[1])

model = YOLO(path_yolo)
names = model.names

fps = int(cap.get(cv2.CAP_PROP_FPS))

Shift_Time = 0
cars = {}
car_id_counter = 0
initial_distance = 0
count = 0
Speed = {}
Count_Violations = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 500))
    frame_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    results = model(frame_roi, stream=True)

    cv2.line(frame, (width_line1_x1, height_line1_y1), (width_line2_x2, height_line2_y2), (255, 0, 255), 5)

    cv2.putText(frame, f"Cars {car_id_counter}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Speed Violations {Count_Violations}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            x = x1
            y = y1+20
            conf = box.conf[0]
            conf = (math.ceil(conf * 100)) / 100
            names_object = names[int(box.cls)]

            if conf > 0.5 and names_object == "car":
                car_id = None
                car_id_speed_violations = None
                for existing_id, (prev_x, prev_y, prev_w, prev_h) in cars.items():
                    distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    distance_used = prev_x + initial_distance
                    if distance < 45:
                        car_id = existing_id
                        break

                if car_id is None:
                    car_id = len(cars) + 1
                    car_id_counter += 1

                cars[car_id] = (x, y, w, h)
                current_time = time.time()
                print(f"{car_id} is detected at {current_time}")

                message = "Violent"
                if len(cars) > 1:
                    speed = SpeedEstimator([prev_x, prev_y], fps)
                    speed_estimate = speed.estimateSpeed()
                    cvzone.putTextRect(frame_roi, f"{speed_estimate} km/h", (x, y - 100), scale=2, thickness=1)
                    Speed_Violation = None
                    if speed_estimate > 40:
                        for existing_id_prev, (prev_x_prev, prev_y_prev, prev_w_prev, prev_h_prev) in Speed.items():
                            distance_speed = np.sqrt((x - prev_x_prev) ** 2 + (y - prev_y_prev) ** 2)
                            distance_used_speed = prev_x_prev + initial_distance
                            if distance_speed < 45:
                                Speed_Violation = existing_id_prev
                                break

                        if Speed_Violation is None:
                            Speed_Violation = len(Speed) + 1
                            Count_Violations += 1

                        Speed[Speed_Violation] = (x, y, w, h)
                        Data_speed = frame_roi[prev_x:prev_y, prev_x+prev_w:prev_y+prev_h]
                        cvzone.cornerRect(frame_roi, (x, y, w, h), l=5, rt=1, colorC=(0,0,255), colorR=(0,0,255))
                        cvzone.cornerRect(frame_roi, (x, y, w, h), l=5, rt=1)
                        cvzone.putTextRect(frame_roi, f"{conf}", (x, y - 70), scale=2, thickness=1)
                        cv2.putText(frame, f"Cars {car_id_counter}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        3)
                        cv2.line(frame, (width_line1_x1, height_line1_y1), (width_line2_x2, height_line2_y2),
                                     (0, 255, 0), 5)
                        cvzone.putTextRect(frame_roi, f"{message}", (x, y-140) ,scale=2, thickness=1)
                        cv2.putText(frame, f"Speed Violations {Count_Violations}", (15,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

                        save = f"{count}.jpg"
                        roi_vehicles = frame_roi[y:y + h, x:x + w]
                        cv2.imwrite(f"Speed_Violents/{save}", roi_vehicles)
                        count += 1

                    else:
                        cvzone.cornerRect(frame_roi, (x, y, w, h), l=5, rt=1)
                        cvzone.putTextRect(frame_roi, f"Car {car_id}", (x, y - 20), scale=2, thickness=1)
                        cvzone.putTextRect(frame_roi, f"{conf}", (x, y - 70), scale=2, thickness=1)
                        cv2.putText(frame, f"Cars {car_id_counter}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                    3)
                        cv2.line(frame, (width_line1_x1, height_line1_y1), (width_line2_x2, height_line2_y2),
                                 (0, 255, 0), 5)
                        cv2.putText(frame, f"Speed Violations {Count_Violations}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Frame_roi", frame_roi)

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
