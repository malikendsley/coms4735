
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
    files = [f for f in os.listdir('img') if f.endswith('.ppm')]
    testing_files = [f for f in os.listdir('testing') if f.endswith('.ppm')]
    
    # get another list in reduced bit depth
    ppms: list[PPMImage] = []
    for f in testing_files if TEST else files:
        ppms.append(PPMImage('testing/' + f, laplace_depth=8))

    #run the hyper tuner to find the best bit depths
    #best_bit_depths = bit_hyper_tuner(crowd_matrix, ppms)
    # shown to be (1, 3, 2)

    #score_images_color(ppms, crowd_matrix)

    for ppm in ppms:
        print("image:", ppm.filename)
        ppm.show_color(40)
        ppm.show_gray(40)
        ppm.show_texture(40)
        print("Texture\n", ppm.texture)
        ppm.laplace_convolve(ppm.grayscale)
if __name__ == '__main__':
    main()