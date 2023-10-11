import cv2
import numpy as np
import time

path = r"C:\Users\doant\Downloads\traffic_-_27260 (Original).mp4"
Detector = cv2.CascadeClassifier(r"C:\Users\doant\PycharmProjects\pythonProject1\venv\Lib\site-packages\cv2\data\haarcascade_car.xml")

cam = cv2.VideoCapture(path)

ret, prev_frame = cam.read()
prev_frame = cv2.resize(prev_frame, (800, 500))

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

original_fps = int(cam.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, original_fps, (800, 500))

count = 1
initial_distance = 0

cars = {}

mask = np.zeros_like(prev_frame)
color = np.random.randint(0, 255, (100, 3))

while True:
    ret, frame = cam.read()

    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    frame_copy = frame.copy()

    fore_ground = bg_subtractor.apply(frame_copy)
    fore_ground = cv2.cvtColor(fore_ground, cv2.COLOR_GRAY2BGR)

    Bitwise = cv2.bitwise_and(frame_copy, fore_ground)

    Car_Detector = Detector.detectMultiScale(frame_copy, 1.25, 3)
    cv2.line(frame, (20, 20), (780, 20), (0, 0, 255), 2)
    cv2.line(frame, (20, 480), (780, 480), (0, 0, 255), 2)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in Car_Detector:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(frame, (x + w//2, y + h//2), 3, (0, 255, 0), -1)
        cv2.line(frame, (x+w//2, 20), (x+w//2, y+h//2), (0,255,0), 2)

        car_id = None

        for existing_id, (prev_x, prev_y, _, _) in cars.items():
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            distance_used = prev_x + initial_distance
            print("distance use", distance_used)
            if distance < 45:
                car_id = existing_id
                break

        if car_id is None:
            car_id = len(cars) + 1

        cars[car_id] = (x, y, w, h)

        current_time = time.time()

        print(f"Car {car_id} detected at time: {current_time}")

        cv2.putText(frame, f"Car {car_id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(f"data_crop/{count}.jpg", frame[y:y + h, x:x + w])
        count += 1

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Frame", Bitwise)

    if cv2.waitKey(1000 // original_fps) & 0xFF == ord('q'):
        break

cam.release()
output.release()
cv2.destroyAllWindows()
