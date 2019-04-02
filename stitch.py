import cv2
import numpy as np
def find_keypoints(step,img):
    row = img.shape[0]
    col = img.shape[1]
    arr = []
    for i in range(0,col,step):
        for j in range(0,row,step):
            keypoint = cv2.KeyPoint(i,j,step)
            arr.append(keypoint)
    return arr

def find_descriptors(img,step,keypoints):
    ans = []
    intensityKeypoints = []
    for i in range(len(keypoints)):
        arr = np.zeros((step,step))
        x1,y1 = keypoints[i].pt
        row = img.shape[0]
        col = img.shape[1]
        if (x1+step <= row and y1+step <= col):
            intensityKeypoints.append(keypoints[i])
            x1,y1 = int(x1),int(y1)
            arr = np.float32(img[x1:x1 + step, y1:y1 + step].flatten())
            mean = np.mean(arr)
            arr = arr-mean
            div = np.sqrt(np.sum(np.multiply(arr,arr)))
            if (div == 1):
                arr = arr/pow(div,0.5)
            ans.append(arr/div)
    return ans,intensityKeypoints

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

    def find_dense_sift_features(self):
        print('Finding Dense sift features for image',self.name)
        self.kp = find_keypoints(15,self.image)
        self.feat = self.feature_finder.compute(self.image,self.kp)

    def find_window_based_correlation_features(self):
        print('Finding window based correlation features for image',self.name)
        keypoints = find_keypoints(5,self.image)
        self.feat,self.kp = find_descriptors(self.image,5,keypoints)
