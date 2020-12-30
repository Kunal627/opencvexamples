import cv2
import os
import numpy as np

class SimpleDataSetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagepaths, verbose=-1):

        data = []
        labels = []

        for (i, path) in enumerate(imagepaths):

            image = cv2.imread(path)
            # /path/to/dataset/{class}/{image}.jpg
            label = path.split(os.path.sep)[-2]

            if self.preprocessors is not None:
#           preprocess images
                for pre in self.preprocessors:
                    image = pre.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagepaths)))

        return (np.array(data), np.array(labels))


