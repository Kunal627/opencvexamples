# use zernikemoments for describing the image and store the index

from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root, root1 = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(root))
sys.path.append(str(root1))

from descriptor.zernikemoments import ZernikeMoments
import argparse
import imutils
from imutils.paths import list_images
import os
import cv2
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sprites", help="path to pokemon images")
parser.add_argument("-i", "--index", help="path to where index will be written")
#parser.add_argument("-z", "--zeradius", help="zernike radius")
args = vars(parser.parse_args())

desc = ZernikeMoments(21)
index = {}

print("[INFO] -- started indexing......")
for spritePath in list_images(args["sprites"]):
    
    name = os.path.basename(spritePath).replace(".png", "")
    print("------------------------------------>", name)
    image = cv2.imread(spritePath)
    # grayscale conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # pad the image with white border
    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)

    # invert the image and threshold
    invert = cv2.bitwise_not(image)
    invert [invert > 0] = 255

    outline = np.zeros(image.shape).astype("uint8")
    cnts = cv2.findContours(invert.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)

    moments = desc.describe(outline)
    index[name] = moments


f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

