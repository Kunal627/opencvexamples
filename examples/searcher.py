from pathlib import Path # if you haven't already done so
import sys, os
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np

class Searcher:
    def __init__(self, index):
        self.index = index
    
    def search(self, queryFeatures):

        results = {}

        for (fname, features) in self.index.items():
            d = self.ch2_distance(features, queryFeatures)
            results[fname] = d

        # sort based on smaller distance first
        results = sorted([(v,k) for (k,v) in results.items()])

        return results


    def ch2_distance(self, histA, histB, e = 1e-10):
        # chi square distance
        d = 0.5 * np.sum([((a - b) ** 2)/ (a + b + e) for (a,b) in zip(histA, histB)])
        return d