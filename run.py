import os
import math
import argparse
from typing import Union, List, Optional, Tuple
import cv2
import numpy as np
import scipy.sparse.csr as csr
import scipy.sparse.csgraph as csgraph
import matplotlib.pyplot as plt
from img_stitch import ImageStitcher
from stitch import _StitchImage
from sklearn.cluster import SpectralClustering

def find_matches(im1, im2, matcher):
    matches = matcher.knnMatch(im1.feat, im2.feat, k=2)
    good = [i for i, j in matches if i.distance < 0.7 * j.distance]
    return len(good), len(matches)

def find_clusters(inputs, num_clusters):
    marking = {}
    for i,image in enumerate(inputs):
        if isinstance(image, str):
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGBA)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        fname = os.path.splitext(os.path.split(image)[1])[0]
        image_f = _StitchImage(img, name=fname)
        # find features
        image_f.find_features()
        marking[i] = image_f
    
    edge_matrix = []
    matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    for i in range(len(marking)):
        temp = []
        for j in range(len(marking)):
            good, matches = find_matches(marking[i], marking[j], matcher)
            if matches == 0:
                temp.append(0)
            else:
                if good > 0.1 * matches:
                    temp.append(1)
                else:
                    temp.append(0)
        edge_matrix.append(temp)
    adjacency_matrix = np.array(edge_matrix)
    w, v = np.linalg.eig(adjacency_matrix)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(w)
    print(v)
    sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=100)
    sc.fit(adjacency_matrix)
    groups = {}
    for i in range(len(sc.labels_)):
        try:
            groups[sc.labels_[i]].append(i)
        except:
            groups[sc.labels_[i]] = [i]
    clusters = []
    for g in groups:
        clusters.append([inputs[x] for x in groups[g]])
    print("@@@@@@@@@@@@@@@@@@@@@@")
    print(clusters)
    # strf = ''
    # for i in edge_matrix:
        # for j in i:
            # strf += str(j) + ' '
        # strf += '\n'
    # print(strf)
    return clusters

def run():
    ''' Driving Function + Argument Handling
    '''
    parser = argparse.ArgumentParser(description='Panoramic Stitch')
    parser.add_argument(
        'input', nargs='+',
        help='The input image files')
    parser.add_argument(
        '-o', required=True, dest='output',
        help='Where to put the resulting stitched image')
    parser.add_argument(
        '-n', required=True, dest='num_clusters',
        help='Expected number of panoramas(clusters) in the image set')
    parser.add_argument(
        '-b', '--base', type=int,
        help='Base Image Index')
    args = parser.parse_args()

    if args.base is not None:
        stitch.center = args.base

    clusters = find_clusters(args.input, int(args.num_clusters)) 
    # Clusters is a list of lists, with each list being a group of pics to be stitched
    for i,group in enumerate(clusters):
        stitch = ImageStitcher()
        for infile in group:
            stitch.add_image(infile)
        try:
            result = stitch.stitch()
            cv2.imwrite(str(i) + "_" + args.output, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        except:
            print("Only One File Input")
    
if __name__ == '__main__':
    run()
