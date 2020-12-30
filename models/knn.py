from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess.simplepreprocessor import SimplePreprocessor
from loader.simpledataloader import SimpleDataSetLoader
from imutils import paths
import argparse


parser = argparse.ArgumentParser(prog='knn')
parser.add_argument("-d", "--dataset", required=True, help="dataset path")
parser.add_argument("-k", "--neighbors", required=True, type=int, default=1, help="dataset path")
parser.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(parser.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# init preprocessor
sp = SimplePreprocessor(32,32)
sdl = SimpleDataSetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0], 3072))
# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
target_names=le.classes_))