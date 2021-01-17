

import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.layers.core import Flatten
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

class DataLoader:
    def __init__(self, inPath, cformat="channels_last"):
        self.inpath = inPath
    
    def load(self):
        train = pd.read_csv(self.inpath + "/train.csv")
        trainY = train['label'].to_numpy()
        train = train.iloc[:, 1:].to_numpy().astype('uint8')
        predinput  = pd.read_csv(self.inpath + "/test.csv")
        predinput = predinput.iloc[:, :].to_numpy().astype('uint8')
        
        if K.image_data_format() == "channels_first":
            trainX = train.reshape(train.shape[0],1,28, 28)
            predinput = predinput.reshape(predinput.shape[0],1,28, 28)
        else:
            trainX = train.reshape(-1,28, 28,1)
            predinput = predinput.reshape(-1,28, 28, 1)
            
        return (trainX, trainY, predinput)
    



class MyCNN:            
    @staticmethod   
    def build(inpshape, classes):
        model = Sequential()
        model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=inpshape))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32,kernel_size=3,activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.4))
        
        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64,kernel_size=3,activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))

#        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu', padding="same", input_shape=inpshape,kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)))
#        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu', padding="same", kernel_regularizer=l2(0.0005)))
#        model.add(BatchNormalization(axis=-1))
#        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#        model.add(Dropout(0.25))
#        
#        
#        model.add(Conv2D(64, (5,5), strides=(1,1), activation = 'relu', padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005)))
#        model.add(Conv2D(64, (5,5), strides=(1,1), activation = 'relu', padding="same", kernel_regularizer=l2(0.0005)))
#        model.add(BatchNormalization(axis=-1))
#        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#        model.add(Dropout(0.25))
#        
#        model.add(Flatten())        
#        model.add(Dense(512, kernel_initializer="he_normal"))
#        model.add(Activation("relu"))
#        model.add(BatchNormalization())
#        model.add(Dropout(0.25))
#        
#        model.add(Dense(256, kernel_initializer="he_normal"))
#        model.add(Activation("relu"))
#        model.add(BatchNormalization())
#        
#        model.add(Dense(128, kernel_initializer="he_normal"))
#        model.add(Activation("relu"))
#        model.add(Dropout(0.25))
#        
#        model.add(Dense(84, kernel_initializer="he_normal"))
#        model.add(Activation("relu"))
#        model.add(Dropout(0.25))
#        
#        model.add(Dense(classes, activation='softmax'))

        return model


# load the input dataset and transform the df to 28 * 28 array
dataloader = DataLoader("/kaggle/input/digit-recognizer/")
(trainX, trainY, predinp) = dataloader.load()
print(trainY.shape, trainX.shape, predinp.shape)
# scale the data [0,1]
trainX = (trainX - trainX.min()) / (trainX.max() - trainX.min())
predinp = (predinp - predinp.min()) / (predinp.max() - predinp.min())


(trainX, testX, trainY, testY) = train_test_split(trainX, trainY.astype("int"), test_size=0.25, random_state=111)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY  = lb.transform(testY)


model = MyCNN.build((28, 28, 1), 10)

opt = SGD(lr=0.001,momentum=0.9)
#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

epochs=35
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=64, epochs=epochs, verbose=1)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

Y = model.predict(predinp)
print(Y[0])

Yhat = np.argmax(Y, 1)
Yhat.shape

Ypred = pd.DataFrame()
#Ypred['ImageId'] = Yhat.reset_index().index
Ypred['ImageId'] = np.arange(Yhat.shape[0]) + 1
Ypred['Label'] = Yhat

print(Ypred)
Ypred.to_csv("submission.csv", index=False)