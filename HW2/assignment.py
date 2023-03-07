
import os
import numpy as np
from image import PPMImage
from util import *

#matrix data is stored in a txt file called Crowd.txt
#a 40x40 matrix of numbers delimited by up to 3 spaces
#load this into memory
crowd_matrix = np.loadtxt('Crowd.txt')

# get all the ppm files in the img folder
files = [f for f in os.listdir('img') if f.endswith('.ppm')]

# get another list in reduced bit depth
ppms = [] #list of PPMImage objects
for i, f in enumerate(files):
    ppms.append(PPMImage('img/' + f, i, (1, 3, 2)))

#run the hyper tuner to find the best bit depths
#best_bit_depths = bit_hyper_tuner(crowd_matrix, ppms)
# shown to be (1, 3, 2)

score_images(ppms, crowd_matrix)

