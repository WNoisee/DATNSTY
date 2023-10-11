import cv2
import numpy as np

path = r"C:\Users\doant\Downloads\traffic_-_27260 (Original).mp4"

# Tạo Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

cam = cv2.VideoCapture(path)
ret, frame_1 = cam.read()
frame_1 = cv2.resize(frame_1, (800, 500))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lucas_kanade_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

prev_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
prev_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
mask = np.zeros_like(frame_1)

while True:
    ret, frame_origin = cam.read()

    if not ret:
        break

    if ret:

        frame_origin = cv2.resize(frame_origin, (800, 500))

        frame_mask = bg_subtractor.apply(frame_origin)

        new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, frame_mask, prev_corners, None,
                                                               **lucas_kanade_params)

        good_new = new_corners[status == 1]
        good_old = prev_corners[status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

        img = cv2.add(frame_origin, mask)

        prev_gray = frame_mask.copy()
        prev_corners = good_new.reshape(-1, 1, 2)

        cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
