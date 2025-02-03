import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.feature
from matplotlib.pyplot import imread
import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage import feature

def getMChist(image,img_gray_int,bins,hist_min,hist_max,dx,da,var1,var2,var3,lev):
    hist = [0] * len(image)
    niv = np.round(256/lev).astype(int)
    for i in range(0, len(image)):
        if len(image[i].shape) < 3:
            g = np.floor(np.divide(img_gray_int[i], lev)).astype(int)
            cooc1 = skimage.feature.graycomatrix(g, distances=[dx[0]], angles=[da[0]], levels=niv, symmetric=False,normed=False)
            cooc2 = skimage.feature.graycomatrix(g, distances=[dx[1]], angles=[da[1]], levels=niv, symmetric=False, normed=False)
            cooc3 = skimage.feature.graycomatrix(g, distances=[dx[2]], angles=[da[2]], levels=niv, symmetric=False,  normed=False)
        else:
            b, g, r = cv2.split(image[i])

            b = np.floor(np.divide(b, lev)).astype(int)
            g = np.floor(np.divide(g, lev)).astype(int)
            r = np.floor(np.divide(r, lev)).astype(int)
            cooc1 = skimage.feature.graycomatrix(b, distances=[dx[0]], angles=[da[0]], levels=niv, symmetric=False, normed=False)
            cooc2 = skimage.feature.graycomatrix(g, distances=[dx[1]], angles=[da[1]], levels=niv, symmetric=False,normed=False)
            cooc3 = skimage.feature.graycomatrix(r, distances=[dx[2]], angles=[da[2]], levels=niv, symmetric=False, normed=False)

        coocS = (var1*cooc1 +var2*cooc2 +var3*cooc3)
        hist[i], _ = np.histogram(coocS, bins, range=(hist_min, hist_max), density=True)

    return hist

def getLBPhist(image,numPoints,radius):
    method_lbp = ['default',
                  'ror',
                  'uniform',
                  'var']
    id_method_lbp = 0
    hist = [0] * len(image)
    for i in range(0,len(image)):
        lbp = feature.local_binary_pattern(image[i], numPoints, radius, method=method_lbp[id_method_lbp])
        hist[i], _ = np.histogram(lbp, density=True, bins=2 ** numPoints, range=(0, 2 ** numPoints))
    return hist


def findBestMatch(hist_query,hist_database,names,name_query,id_method_distance):
    min_distance = [99999, 99999, 99999];
    min_file = ["", "", ""];
    i = 0
    method_distance = [cv2.HISTCMP_CORREL,
                       cv2.HISTCMP_CHISQR,
                       cv2.HISTCMP_INTERSECT,
                       cv2.HISTCMP_BHATTACHARYYA,
                       cv2.HISTCMP_HELLINGER,
                       cv2.HISTCMP_CHISQR_ALT,
                       cv2.HISTCMP_KL_DIV]
    for j in range(0, len(hist_database)):

        if(id_method_distance == 7):
            distance = np.linalg.norm(hist_query - hist_database[j])
        else:
            distance = cv2.compareHist(hist_query.astype(np.float32),  hist_database[j].astype(np.float32), method_distance[id_method_distance])

        if (id_method_distance == 0 or id_method_distance == 2):
            distance = 1/distance # la corrélation et l'intersection donne 1 si plus proche et 0 si plus loin

        max_distance = max(min_distance)
        if (distance < max_distance):
            ind = min_distance.index(max_distance)
            min_distance[ind] = distance
            min_file[ind] = names[j].split("\\")[-1].split("_")[0]

    results = min_file.count(name_query.split("_")[0])
    return results

def readFiles():
    i = 0
    directory = 'data'
    names_query = [""] * 9
    image_query = [0] * 9
    image_query_gray = [0] * 9
    for query in os.listdir(directory):
        if query == "database":
            continue
        names_query[i] = query
        image_query[i] = mpimg.imread("data/" + query)
        image_query_gray[i] = cv2.imread("data/" + query, cv2.IMREAD_GRAYSCALE)
        i += 1

    image_database = [0] * 40
    image_database_gray = [0] * 40
    names = [""] * 40
    results = [0] * 8
    i = 0
    directory = 'data/database'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            names[i] = repr(f)
            image_database[i] = mpimg.imread(f)
            image_database_gray[i] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            i += 1
    return image_query, image_database,image_query_gray,image_database_gray, names,names_query