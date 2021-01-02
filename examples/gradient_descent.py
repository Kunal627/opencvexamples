# implemented mini batch gradient descent
# batchsize = 1 for pure SGD
# for mini batch use batchsize in power of 2

from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):

    return 1.0/ (1 + np.exp(-x))


def predict(X, W):
    pred = sigmoid_activation(X.dot(W))
    pred[pred <= 0.5] = 0
    pred[pred > 0.5]  = 1

    return pred

def next_batch(X, y, batchSize):

# loop over our dataset ‘X‘ in mini-batches, yielding a tuple of
# the current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])


parser = argparse.ArgumentParser(prog='gradient')
parser.add_argument("-e", "--epochs", type=float, default=100, help="# epochs ")
parser.add_argument("-a", "--alpha", type=float, default=.01, help="learning rate")
parser.add_argument("-b", "--batch-size", type=int, default=32,help="size of SGD mini-batches")
args = vars(parser.parse_args())

(X, y) = make_blobs(n_samples=1024, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# add column for bias and initialize it with 1
X = np.c_[X, np.ones((X.shape[0]))]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training...")
#initialize weights
W = np.random.randn(X.shape[1], 1)
losses = []

# loop for epochs
for epoch in np.arange(0, args["epochs"]):

    epochLoss = []
    # loop over our data in batches
    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
        preds = sigmoid_activation(batchX.dot(W))
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))
        gradient = batchX.T.dot(error)
        W += -args["alpha"] * gradient
    
    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()