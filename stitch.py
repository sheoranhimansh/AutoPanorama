import cv2

class _StitchImage:
    _lastIdx = 1

    def __init__(self, image, name: str=None):
        self.image = image
        self.kp = None
        self.feat = None
        self.feature_finder = cv2.xfeatures2d.SIFT_create()

        if name is None:
            name = '%02d' % (_StitchImage._lastIdx)
            _StitchImage._lastIdx += 1
        self.name = name

    def find_features(self):
        print('Finding features for image', self.name)
        self.kp, self.feat = self.feature_finder.detectAndCompute(self.image, None)


