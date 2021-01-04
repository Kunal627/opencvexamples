from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from loader.simpledatasetloader import SimpleDataSetLoader
from preprocess.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocess.simplepreprocessor import SimplePreprocessor

sdl = SimpleDataSetLoader(preprocessors=None)