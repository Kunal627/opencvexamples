from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# import the necessary packages
from examples.lenet import LeNet
#from keras.utils import plot_model
import keras
#import pydotplus
import pydot
from keras.utils.vis_utils import plot_model
keras.utils.vis_utils.pydot = pydot
# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)