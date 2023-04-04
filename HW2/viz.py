# this file will focus on visualizing the data in a presentable way

import matplotlib.pyplot as plt
import numpy as np

from util import score_images_color



#all steps will have the same visualization process
#each function will be given a list of PPMS, a selection matrix, a score, and a personal score
#using the selection matrix, the function will display the images that were selected
#each image will be labelled with q + the image number, each image will also be labelled with the score it received
#each row will have a score label, and the total score will be displayed at the bottom
#the happiness score will also be displayed at the bottom
#load the crowd matrix
crowd_matrix = np.loadtxt('Crowd.txt')
sparse_matrix = np.loadtxt('Sparse.txt')
personal_matrix = np.loadtxt('MyPreferences.txt')
#delete leftmost column from personal matrix
personal_matrix = np.delete(personal_matrix, 0, 1)

def visualize_color_step(ppms, crowd_matrix, personal_matrix, scoring_function: function):
    #get the score, selections, and personal score
    score, selections, personal_score = scoring_function(ppms, crowd_matrix, personal_matrix)


    #for each row in the selection matrix
    for i, row in enumerate(selections):
        #id is the query image number
        q = ppms[i]
        t1 = ppms[row[0]]
        t2 = ppms[row[1]]
        t3 = ppms[row[2]]
        t1_score = crowd_matrix[i][row[0]]
        t2_score = crowd_matrix[i][row[1]]
        t3_score = crowd_matrix[i][row[2]]
        rowscore = t1_score + t2_score + t3_score
        #TODO display the images, with the labels individual scores underneath and the row score out to the side
    # annotate the bottom with the total score and the personal score

visualize_color_step(ppms, crowd_matrix, personal_matrix, score_images_color)