
import os
from typing import List
import numpy as np
from image import PPMImage
from util import *
import sys

TEST = False

def main():
    # if --test is passed use the testing folder
    if '--test' in sys.argv:
        print('using testing folder')
        global TEST
        TEST = True
    folder = 'testing' if TEST else 'img'
    
    #load the crowd matrix
    crowd_matrix = np.loadtxt('Crowd.txt')

    # get all the ppm files in the desired folder
    print("Finding images...", end="")
    files = [f for f in os.listdir('img') if f.endswith('.ppm')]
    testing_files = [f for f in os.listdir('testing') if f.endswith('.ppm')]
    print("done")
    
    # load all the images, can take a while so print names as they are loaded, 5 per line
    print("Loading images...")
    ppms: List[PPMImage] = []
    for i, f in enumerate(testing_files if TEST else files):
        print(f, end=" | ")
        if (i + 1) % 5 == 0:
            print()
        # these are precomputed hyper parameters for the best gestalt score
        ppms.append(PPMImage(folder + f, (1, 3, 2), 9, 79, 149))
    print("\ndone")
    
    # ======================  Note  ====================== #
    
    # these are all commented out because they are time consuming to run
    # they were all run and the results are shown in the comments and in the report
    # the hyper tuner functions are also commented out because they only need to be run once
    
    # ====================== Step 1 ====================== #
    
    # # run the hyper tuner to find the best bit depths for the color
    # best_bit_depths = color_hyper_tuner(crowd_matrix, ppms)
    # print(best_bit_depths)
    
    ############################################
    # found to be (1, 3, 2) with score 11321.0 #
    ############################################

    # # get the best score and system selections
    # score, selections = score_images_color(ppms, crowd_matrix)

    # ====================== Step 2 ====================== #

    # # run the hyper tuner to find the best bit depth for the texture
    # best_texture_bit_depth = texture_hyper_tuner(crowd_matrix, ppms)
    # print(best_texture_bit_depth)
    
    ######################################
    # shown to be 9 with score of 6836.0 #
    ######################################
    
    # # get the best score and system selections
    # score, selections = score_images_texture(ppms, crowd_matrix)

    # ====================== Step 3 ====================== #

    # # run the hyper tuner to find the best threshold for the binary images
    # best_binary_threshold = shape_hyper_tuner(crowd_matrix, ppms)
    # print(best_binary_threshold)
    
    #####################################
    # shown to be 79 with score of 6563 #
    #####################################
    
    # # get the best score and system selections
    # score, selections = score_images_shape(ppms, crowd_matrix)

    # ====================== Step 4 ====================== #
    
    # # run the hyper tuner to find the best threshold for the binary images
    # best_symmetry_threshold = symmetry_hyper_tuner(crowd_matrix, ppms)
    # print(best_symmetry_threshold)
    
    ######################################
    # shown to be 149 with score of 4711 #
    ######################################
    
    # # get the best score and system selections
    # score, matrix = score_images_symmetry(ppms, crowd_matrix)
    
    # ====================== Step 5 ====================== #
    
    # gestalt score, gathered by weighting the scores of the previous steps
    # since the various scores all range from 0 to 1
    
    # # (hyper) hyper tune the gestalt weights to find the best weights
    # # careful about high resolution, it takes exponentially longer (.01 takes a few hours)
    # best_weights = gestalt_hyper_tuner(crowd_matrix, ppms, resolution=0.01)

    ##############################################################################
    # at resolution 0.1, the best weights are [0.4, 0.2, 0.4, 0.0] at 12469      #
    # at resolution 0.05, the best weights are [0.40, 0.20, 0.40, 0.00] at 12469 #
    # at resolution 0.01, the best weights are [.35, 0.22, 0.43, 0.0] at 12600   #
    ##############################################################################
    
    #score, selections = score_images_gestalt(ppms, crowd_matrix, [0.35, 0.22, 0.43, 0.0])
    
if __name__ == '__main__':
    main()