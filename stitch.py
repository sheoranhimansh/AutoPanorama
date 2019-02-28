import os
import math
import logging
import argparse
from typing import Union, List, Optional, Tuple

import cv2
import numpy as np
import scipy.sparse.csr as csr
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt

class _StitchImage:
    _lastIdx = 1

    def __init__(self, image, name: str=None):
        self.image = image
        self.kp = None
        self.feat = None
        try:
            self.feature_finder = cv2.xfeatures2d.SIFT_create()
            matcher = cv2.BFMatcher_create(cv2.NORM_L2)
        except AttributeError:
            print("Install Sift")

        if name is None:
            name = '%02d' % (_StitchImage._lastIdx)
            _StitchImage._lastIdx += 1
        self.name = name

    def find_features(self):
        print('Finding features for image', self.name)
        self.kp, self.feat = self.feature_finder.detectAndCompute(self.image, None)


