import numpy as np
import cv2

bodydetection = cv2.CascadeClassifier('haarcascade_fullbody.xml')
img = cv2.imread('J.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
body = bodydetection.detectMultiScale(gray, 1.009, 5)

for x, y, w, h in body:
   # so we slightly shrink the rectangles to get a nicer output.
   pad_w, pad_h = int(0.15*w), int(0.02*h)
   cv2.rectangle(img, (x+pad_w+10, y+pad_h+10), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 2)

cv2.imshow('img',img)

crop_img = img[x:x+w, y:y+h]
cv2.imshow('crop',crop_img)

cv2.waitKey(0)
cv2.destroyAllWindows()