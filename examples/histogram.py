from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


import cv2
from matplotlib import pyplot as plt 
import numpy as np

image = cv2.imread(r'./testdata/jp.jpeg')

# get the channels from the BGR image
channels = cv2.split(image)
col = ["b", "g", "r"]

# convert to grayscale
gscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image" , gscale)
cv2.waitKey(0)

plt.figure()
plt.title("image histogram")
plt.xlabel("bins")
plt.ylabel("# pixels in a bin")
feature = []

for (ch, color) in zip (channels, col):
    # create a histogram for each channel and append it to a list
    hist = cv2.calcHist([ch], [0], None, [256], [0,256])
    feature.extend(hist)
    #plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

plt.show()
 
print ("flattened feature vector size: %d", np.array(feature).flatten().shape)

print("=======================channel shape============",channels[1].shape)
# 2d histograms

fig = plt.figure()
# plot a 2D color histogram for green and blue
ax = fig.add_subplot(131)
hist = cv2.calcHist([channels[1], channels[0]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color hist for Green and Blue")
plt.colorbar(p)


# plot a 2D color histogram for green and red
ax = fig.add_subplot(132)
hist = cv2.calcHist([channels[1], channels[2]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color hist for Green and Red")
plt.colorbar(p)


# plot a 2D color histogram for blue and red
ax = fig.add_subplot(133)
hist = cv2.calcHist([channels[0], channels[2]], [0,1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation = "nearest")
ax.set_title("2D Color hist for Blue and Red")
plt.colorbar(p)

plt.show()

# finally, let's examine the dimensionality of one of
# the 2D histograms
print ("2D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))

# 3 d histogram , can't visualize
hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print ("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))