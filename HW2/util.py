import sys
from image import PPMImage
import numpy as np
import math
import typing

def color_distance(img1: PPMImage, img2: PPMImage):
    #ensure images have same bit depth and dimensions
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.color_histogram - img2.color_histogram)) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)

def texture_distance(img1: PPMImage, img2: PPMImage):
    #ensure images have same bit depth and dimensions
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.texture_histogram - img2.texture_histogram)) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)

    
def shape_distance(img1: PPMImage, img2: PPMImage):
    #ensure images have same bit depth and dimensions
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    #xnor the images and sum the result, normalized
    return np.sum(np.logical_xor(img1.binary_for_shape, img2.binary_for_shape)).astype(np.float32) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)
    
    
def score_images_color(ppms, crowd_matrix):
    # make the 40x40 matrix of the L1 distances between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = color_distance(ppms[i], ppms[j])
    
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    return np.sum(crowd3)

# tries to find the best bit depth for each channel by computing all combinations
def color_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    print("Finding best color bit depths...")
    # a better score is lower
    best = -math.inf
    best_bit_depths = (1, 1, 1)
    # iterate from 111 to 666
    for r in range(1, 7):
        for g in range(1, 7):
            for b in range(1, 7):
                # faster than creating a new list each time
                for ppm in ppms:
                    ppm.set_color_depth((r, g, b))
                score = score_images_color(ppms, crowd_matrix)
                if score > best:
                    best = score
                    best_bit_depths = (r, g, b)
                    print(f'new best: {best_bit_depths} with score {best}')
    print(f'Best color bit depths: {best_bit_depths}')
    return best_bit_depths

def score_images_texture(ppms, crowd_matrix):
    # make the 40x40 matrix of the L1 distances between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = texture_distance(ppms[i], ppms[j])
    
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    return np.sum(crowd3)

def texture_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    print("Finding best texture bit depths...")
    # a better score is lower
    best = -math.inf
    best_bit_depth = 1
    # iterate from 1-8 bit grayscale
    for d in range(1, 12):
        # faster than creating a new list each time
        for ppm in ppms:
            ppm.set_laplace_depth(d)
        score = score_images_texture(ppms, crowd_matrix)
        if score > best:
            best = score
            best_bit_depth = d
            print(f'new best: {best_bit_depth} with score {best}')
    print(f'Best texture bit depths: {best_bit_depth}')
    return best_bit_depth

def score_images_shape(ppms, crowd_matrix):
    # make the 40x40 matrix of the L1 distances between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = shape_distance(ppms[i], ppms[j])

    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    return np.sum(crowd3)

# tries to find the best threshold for the binary images by computing all combinations
def shape_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    print("Finding best shape bit depths...")
    # a better score is lower
    best = -math.inf
    threshold = 1
    # iterate from threshold of 1 to 255
    for t in range(1, 256):
        # faster than creating a new list each time
        for ppm in ppms:
            ppm.set_binary_threshold(t)
        score = score_images_shape(ppms, crowd_matrix)
        if score > best:
            best = score
            threshold = t
            print(f'new best: {threshold} with score {best}')
    return threshold