from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import cv2
import imutils

image = cv2.imread(r'./testdata/jp.jpeg')
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# print individual pixel
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

#resize
resized = cv2.resize(image, (200, 200))
#cv2.imshow("Fixed Resizing", resized)
#cv2.waitKey(0)
#(h, w, d) = resized.shape
#print("width={}, height={}, depth={}".format(w, h, d))

r = 300/1366      # new width/ old width
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
#resized = imutils.resize(image, width=300)  -- use imutils to preserve aspect ratio without explicit calculation
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)