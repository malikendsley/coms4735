from image import PPMImage
import numpy as np
import math


def l1_color_distance(img1: PPMImage, img2: PPMImage):
    """Calculate the L1 distance between two images"""
    
    #ensure images have same bit depth and dimensions
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.image.shape != img2.image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.color_histogram - img2.color_histogram)) / (img1.image.shape[0] * img1.image.shape[1] * 2)

def l1_texture_distance(img1: PPMImage, img2: PPMImage):
    """Calculate the L1 distance between two images"""
    
    #ensure images have same bit depth and dimensions
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.image.shape != img2.image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.texture_histogram - img2.texture_histogram)) / (img1.image.shape[0] * img1.image.shape[1] * 2)

def score_images_color(ppms, crowd_matrix):
    # make the 40x40 matrix of the L1 distances between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = l1_color_distance(ppms[i], ppms[j])
    
    # find the 3 most similar image indices for each image
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    
    # use the top3 matrix as a lookup table to get the crowd matrix values
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    #stack a 1-40 array to the top3 matrix so that each row has an index at the beginning
    top3 = np.hstack((np.arange(1, len(ppms) + 1).reshape(len(ppms), 1), top3 + 1))
    print(top3)
    return np.sum(crowd3)

# tries to find the best bit depth for each channel by computing all combinations
def bit_hyper_tuner(crowd_matrix, ppms):
    # a better score is lower
    best = -math.inf
    best_bit_depths = (1, 1, 1)
    # iterate from 111 to 666
    for r in range(1, 7):
        for g in range(1, 7):
            for b in range(1, 7):
                # set the bit depths
                for ppm in ppms:
                    ppm.set_bit_depths((r, g, b))
                # get the distance matrix
                score = score_images_color(ppms, crowd_matrix)
                # keep best score
                if score > best:
                    best = score
                    best_bit_depths = (r, g, b)
                    print(f'new best: {best_bit_depths} with score {best}')
                else:
                    print(f'discarded: {r, g, b}')
    print(f'Best bit depths: {best_bit_depths}')
    return best_bit_depths

def score_images_texture(ppms, crowd_matrix):
    # make the 40x40 matrix of the L1 distances between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = l1_texture_distance(ppms[i], ppms[j])
    
    # find the 3 most similar image indices for each image
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    
    # use the top3 matrix as a lookup table to get the crowd matrix values
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    #stack a 1-40 array to the top3 matrix so that each row has an index at the beginning
    top3 = np.hstack((np.arange(1, len(ppms) + 1).reshape(len(ppms), 1), top3 + 1))
    print(top3)
    return np.sum(crowd3)