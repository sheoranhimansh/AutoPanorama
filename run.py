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
from img_stitch import ImageStitcher
from stitch import _StitchImage

def find_clusters(inputs):
    return [inputs, inputs]

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
        '-b', '--base', type=int,
        help='Base Image Index')
    args = parser.parse_args()

    if args.base is not None:
        stitch.center = args.base

    clusters = find_clusters(args.input) 
    # Clusters is a list of lists, with each list being a group of pics to be stitched
    for i,group in enumerate(clusters):
        stitch = ImageStitcher()
        for infile in group:
            stitch.add_image(infile)
        result = stitch.stitch()
        cv2.imwrite(str(i) + "_" + args.output, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))

    # for infile in args.input:
        # stitch.add_image(infile)
    
if __name__ == '__main__':
    run()
