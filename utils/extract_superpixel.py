import os
from cv2 import data
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# graph
from skimage.future import graph
from skimage.color import gray2rgb
import networkx as nx


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import tqdm
from functools import partial

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to nyu data',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted segment',
                    required=True)
args = parser.parse_args()

data_path = args.data_path
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def extract_superpixel(filepath):
    foldername, filename = filepath.split()
    filename = os.path.join(data_path, foldername, '0', 'left_rgb', filename)
    image = cv2.imread(filename, -1)

    image = img_as_float(image)

    # gradient = sobel(rgb2gray(image))
    # segment = watershed(gradient, markers=m, compactness=0.001)
    segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)

    return segment.astype(np.int16)

def images2seg(filepath):
    foldername, filename = filepath.split()
    segment = extract_superpixel(filepath)
    output_path = os.path.join(output_dir, foldername, '0', 'left_rgb', 
        "{}.npy".format(os.path.splitext(filename)[0]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, segment)
             


# multi processing fitting
executor = ProcessPoolExecutor(max_workers=cpu_count())
futures = []

scenes =  os.listdir(data_path)
all_files = ['{} {}'.format(scene, filename) for scene in scenes 
    for filename in os.listdir(os.path.join(data_path, scene, '0', 'left_rgb'))]

for fp in all_files:
    task = partial(images2seg, fp)
    futures.append(executor.submit(task))

results = []
[results.append(future.result()) for future in tqdm.tqdm(futures)]