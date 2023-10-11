import cv2
import os
import matplotlib.pyplot as plt

path = "Speed_Violents"

for folder in os.listdir(path):
    file = os.path.join(path, folder)
    image = cv2.imread(file)

    img = cv2.selectROI("Image", image)
    crop = image[int(img[1]):int(img[1] + img[3]), int(img[0]):int(img[0]+img[2])]

    blur = cv2.GaussianBlur(crop, (7,7), 1)

    cv2.imshow("img", img)

    cv2.waitKey(0)