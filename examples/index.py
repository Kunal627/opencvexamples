from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from rgbhistogram import RGBHistogram
from imutils.paths import list_images
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# key is file name value is feature vector
index = {}

# 8 bins per channel
desc = RGBHistogram([8,8,8])

for imagePath in list_images(args["dataset"]):
    fname = imagePath[imagePath.rfind("/") + 1 :]

    image = cv2.imread(imagePath)
    features = desc.describe(image)
    index[fname] = features

# index to disk
f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

# show how many images we indexed
print("[INFO] done...indexed {} images".format(len(index)))