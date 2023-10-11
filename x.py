import cv2

path_cascade = r"C:\Users\doant\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\dataXML\haarcascade_russian_plate_number.xml"
path = r"C:\Users\doant\Downloads\pexels-taryn-elliott-5309381 (1080p).mp4"

License_Detector = cv2.CascadeClassifier(path_cascade)

cap = cv2.VideoCapture(path)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 500))

    Detect = License_Detector.detectMultiScale(frame, scaleFactor=1.4, minNeighbors=3)
    for (x,y,w,h) in Detect:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("frame", frame)

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()