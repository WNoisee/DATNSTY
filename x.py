import numpy as np
import cv2

cap = cv2.VideoCapture(r"C:\Users\doant\Downloads\traffic_-_27260 (Original).mp4")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_frame = cv2.resize(old_frame, (800, 500))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,
                             **feature_params)

mask = np.zeros_like(old_frame)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 500))

    if not ret:
        break

    ground = bg_subtractor.apply(frame)
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           p0, None,
                                           **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)

        frame = cv2.circle(frame, (a, b), 5,
                           color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    key = cv2.waitKey(30)
    if key == 113:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)


    cv2.imshow('frame', img)
    cv2.imshow("old_img", mask)
    cv2.imshow("frame_ori", ground)

cap.release()
cv2.destroyAllWindows()
