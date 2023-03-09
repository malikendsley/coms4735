
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
    folder = 'testing/' if TEST else 'img/'
    
    #load the crowd matrix
    crowd_matrix = np.loadtxt('Crowd.txt')
    sparse_matrix = np.loadtxt('Sparse.txt')
    personal_matrix = np.loadtxt('MyPreferences.txt')
    #delete leftmost column from personal matrix
    personal_matrix = np.delete(personal_matrix, 0, 1)
    
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

    # get the best score and system selections
    # score, selections, personal_score = score_images_color(ppms, crowd_matrix, personal_matrix)


    # ====================== Step 2 ====================== #

    # # run the hyper tuner to find the best bit depth for the texture
    # best_texture_bit_depth = texture_hyper_tuner(crowd_matrix, ppms)
    # print(best_texture_bit_depth)
    
    ######################################
    # shown to be 9 with score of 6836.0 #
    ######################################
    
    # # get the best score and system selections
    # score, selections, personal_score = score_images_texture(ppms, crowd_matrix, personal_matrix)

    # ====================== Step 3 ====================== #

    # # run the hyper tuner to find the best threshold for the binary images
    # best_binary_threshold = shape_hyper_tuner(crowd_matrix, ppms)
    # print(best_binary_threshold)
    
    #####################################
    # shown to be 79 with score of 6563 #
    #####################################
    
    # # get the best score and system selections
    # score, selections, personal_score = score_images_shape(ppms, crowd_matrix, personal_matrix)

    # ====================== Step 4 ====================== #
    
    # # run the hyper tuner to find the best threshold for the binary images
    # best_symmetry_threshold = symmetry_hyper_tuner(crowd_matrix, ppms)
    # print(best_symmetry_threshold)
    
    ######################################
    # shown to be 149 with score of 4711 #
    ######################################
    
    # # get the best score and system selections
    # score, selections, personal_score = score_images_symmetry(ppms, crowd_matrix, personal_matrix)
    
    # ====================== Step 5 ====================== #
    
    # gestalt score, gathered by weighting the scores of the previous steps
    # since the various scores all range from 0 to 1
    
    # (hyper) hyper tune the gestalt weights to find the best weights
    # careful about high resolution, it takes exponentially longer (.01 takes a few hours)
    # best_weights = gestalt_hyper_tuner(crowd_matrix, ppms, resolution=0.02)

    ##############################################################################
    # at resolution 0.1, the best weights are [0.5, 0.1, 0.4, 0.0] at 12297      #
    # at resolution 0.05, the best weights are [0.5, 0.25, 0.25, 0.0] at 12469   #
    # at resolution 0.02, the best weights are [0.46, 0.28, 0.24, 0.02] at 12547 #
    ##############################################################################
    
    #score, selections, personal_score = score_images_gestalt(ppms, crowd_matrix, [0.46, 0.28, 0.24, 0.02], personal_matrix)
    
    # ============ Testing Personal Happiness ============= #

    # # test the personal score, since it was added last
    # _, _, personal_score_color = score_images_color(ppms, crowd_matrix, personal_matrix)
    # _, _, personal_score_texture = score_images_texture(ppms, crowd_matrix, personal_matrix)
    # _, _, personal_score_shape = score_images_shape(ppms, crowd_matrix, personal_matrix)
    # _, _, personal_score_symmetry = score_images_symmetry(ppms, crowd_matrix, personal_matrix)
    # score, _, personal_score_gestalt = score_images_gestalt(ppms, crowd_matrix, [0.46, 0.28, 0.24, 0.02], personal_matrix)
    
    # print(f'personal score color: {personal_score_color}')
    # print(f'personal score texture: {personal_score_texture}')
    # print(f'personal score shape: {personal_score_shape}')
    # print(f'personal score symmetry: {personal_score_symmetry}')
    # print(f'personal score gestalt: {personal_score_gestalt}')
    # # ensure the gestalt score is unchanged from above
    # print(f'gestalt score: {score}')
    
    # ======================  Step 6 ====================== #
    
    # # essentially redo step 5 but using sparse_matrix instead of crowd_matrix
    # # tune the weight on the sparse matrix so they match my preferences
    # best_weights = gestalt_hyper_tuner(sparse_matrix, ppms, resolution=0.05)
    # # check the new personal score
    
    #######################################################################
    # tuning on personal sparse matrix with resolution 0.05               #
    # gives best weights = [0.15, 0.25, 0.6, 0.0] with score 125 (52.08%) #
    # no time for 0.01 but likely a little better                         #
    #######################################################################
    
    # _, _, personal_score_gestalt = score_images_gestalt(ppms, crowd_matrix,[0.5, 0.25, 0.25, 0.0], personal_matrix)
    # score, _, _ = score_images_gestalt(ppms, sparse_matrix, [0.15, 0.25, 0.6, 0.0], personal_matrix)
    # print(f'old personal score: {personal_score_gestalt} out of 120 ({personal_score_gestalt / 120 * 100}%)')
    # print(f'new personal score: {score} out of 240 ({score / 240 * 100}%)')
    
if __name__ == '__main__':
    main()