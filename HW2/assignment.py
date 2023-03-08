
import os
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


    #matrix data is stored in a txt file called Crowd.txt
    #a 40x40 matrix of numbers delimited by up to 3 spaces
    #load this into memory
    crowd_matrix = np.loadtxt('Crowd.txt')

    # get all the ppm files in the img folder
    print("Loading images...", end="")
    files = [f for f in os.listdir('img') if f.endswith('.ppm')]
    testing_files = [f for f in os.listdir('testing') if f.endswith('.ppm')]
    print("done")
    
    print("Loading images...")
    # get another list in reduced bit depth
    ppms: list[PPMImage] = []
    for i, f in enumerate(testing_files if TEST else files):
        print(f, end=" | ")
        if (i + 1) % 5 == 0:
            print()
        ppms.append(PPMImage('img/' + f, (1, 3, 2), 9))
    print("\ndone")
    
    # ====================== Step 1 ====================== #
    
    #run the hyper tuner to find the best bit depths for the color
    # best_bit_depths = color_hyper_tuner(crowd_matrix, ppms)
    # found to be (1, 3, 2) with score 11321.0

    #score_images_color(ppms, crowd_matrix)

    # ====================== Step 2 ====================== #

    # run the hyper tuner to find the best bit depth for the texture
    # best_texture_bit_depth = texture_hyper_tuner(crowd_matrix, ppms)
    # shown to be 9 with score of 6836.0

    score_images_texture(ppms, crowd_matrix)

    # ====================== Step 3 ====================== #

if __name__ == '__main__':
    main()