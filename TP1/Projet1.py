# Code to test final results

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.feature
from matplotlib.pyplot import imread
import cv2
import numpy as np
from pylab import ginput
from scipy import signal
from scipy import stats
import matplotlib
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from skimage import data
import statistics
from methodsProj1 import *
import time

current_working_directory = os.getcwd()
print(current_working_directory)

bins = 256
hist_min = 0
numPoints = 20
radius = 3
results = [0] * 8
image_query, image_database,image_query_gray,image_database_gray, names,names_query = readFiles()
textures = ['airplane', 'ball', 'car', 'cat', 'dolphin', 'face', 'lotus', 'pickles']
best_j = 0
best_k = 0
best_i = 0
best_score = 0
data_x = []
data_y = []
time_y = []
dx = [3, 3 , 3]
da = [np.pi/3,5*np.pi/6,5*np.pi/6]
niveaux = [1,2,4,8,16,32,64,128]
levels = niveaux[0]
hist_max = np.round(256/levels)
method_score = [0] * 8
method_names = ["Correlation","Chi-Square","Intersection","Bhattacharyya","Hellinger","Chi-Square Alt","Kullback-Leibler","L2"]

# TO TEST WITH BOUNDING BOXES:
# Change all data/ to data_box is the readFiles function

for i in range (0,8):
        ts = time.time()
        hist_query = getMChist(image_query, image_query_gray, bins, hist_min, hist_max, dx, da, 1, 1, 8, levels)
        hist_database = getMChist(image_database, image_database_gray, bins, hist_min, hist_max, dx, da, 1, 1, 8,levels)
        for N in range(0, len(hist_query) - 1):
                results[N] = findBestMatch(hist_query[N], hist_database, names, names_query[N], i)
        score = np.average(results) * 100 / 3
        data_x.append(i)
        data_y.append((score))
        te = time.time()
        time_y.append(te - ts)

plt.bar(textures, results)
plt.xlabel('Results by query for L2 comparaison')
plt.ylabel('Score (%)')
plt.title('Best method for MC')
plt.show()

plt.bar(method_names, data_y)
plt.xlabel('Average result by comparaison method')
plt.ylabel('Score (%)')
plt.title('Best method for MC')
plt.show()

for i in range (0,8):
        ts = time.time()
        hist_query = getLBPhist(image_query_gray, numPoints, radius)
        hist_database = getLBPhist(image_database_gray, numPoints, radius)
        for N in range(0, len(hist_query) - 1):
                results[N] = findBestMatch(hist_query[N], hist_database, names, names_query[N], i)
        score = np.average(results) * 100 / 3
        data_x.append(i)
        data_y.append((score))
        te = time.time()
        time_y.append(te - ts)

plt.bar(textures, results)
plt.xlabel('Results by query for L2 comparaison')
plt.ylabel('Score (%)')
plt.title('Best method for LBP')
plt.show()

plt.bar(method_names, data_y)
plt.xlabel('Average result by comparaison method')
plt.ylabel('Score (%)')
plt.title('Best method for LBP')
plt.show()

