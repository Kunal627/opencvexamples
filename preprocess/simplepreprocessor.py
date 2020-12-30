import cv2

# this preprocessor resizes the imgage without maintaining the aspect ratio
class SimplePreprocessor:

    def __init__(self, height, width, inter=cv2.INTER_AREA):

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):

        return cv2.resize(image,(self.width, self.height), interpolation=self.inter)

