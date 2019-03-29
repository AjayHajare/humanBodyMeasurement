import cv2
import numpy as np

img = cv2.imread("B.jpg")
cv2.imshow('original', img)

can = cv2.Canny(img, 100, 200)
cv2.imshow('Canny', can)

print(can.shape)

#####################  FOR LOOP TO CALCULATE PIXELS

cv2.waitKey(0)
cv2.destroyAllWindows()

