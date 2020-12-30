from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import cv2
import imutils
import pickle
from imutils.paths import list_images

class RGBHistogram:

    def __init__(self,bins):
        self.bins = bins

    def describe(self, image):

        # calculate the 3d histogram
        hist = cv2.calcHist([image], [0,1,2], None, self.bins ,[0, 256, 0, 256, 0, 256])

        # normalize for scaled up or scaled down images to get almost similar histograms

        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        
        else:
            hist = cv2.normalize(hist,hist)
        
        return hist.flatten()

