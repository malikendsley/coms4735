
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from image import PPMImage
from util import *
import sys
from PIL import Image, ImageDraw, ImageFont

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
    
    # ======================  Visualization ====================== #
    # #for all ppms exchange RGB to BGR
    font = ImageFont.truetype('arial.ttf', size=16)
    for ppm in ppms:
        ppm.original_image = ppm.original_image[:, :, ::-1]
    def visualize_images(ppms, crowd_matrix, personal_matrix, score_function):
    #get the score, selections, and personal score
        if score_function.__name__ == 'score_images_gestalt':
            score, selections, p_score = score_function(ppms, crowd_matrix, [0.46, 0.28, 0.24, 0.02], personal_matrix)
        else:
            score, selections, p_score = score_function(ppms, crowd_matrix, personal_matrix)

        #initialize an image in PIL
        image = Image.new('RGB', (850, 56 * len(ppms)))
        #annotate the top of the picture with the total score and the personal score
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), f'total: {int(score)} out of 18021 ({score / 18021.0 * 100:.2f}%)', font=font)
        draw.text((600, 0), f'happiness: {p_score} out of {len(ppms) * 3} ({p_score / (len(ppms) * 3) * 100:.2f}%)', font=font)
        #put "scoring_function - based selections" at the top of the image
        draw.text((300, 0), f'{score_function.__name__} - based selections', (255, 255, 255))
        #for each row in the selection matrix
        for i, row in enumerate(selections):
            #id is the query image number
            q = ppms[i]
            t1 = ppms[row[0] - 1]
            t2 = ppms[row[1] - 1]
            t3 = ppms[row[2] - 1]
            t1_score = crowd_matrix[i][row[0] - 1].astype(int)
            t2_score = crowd_matrix[i][row[1] - 1].astype(int)
            t3_score = crowd_matrix[i][row[2] - 1].astype(int)
            rowscore = int(t1_score + t2_score + t3_score)
            #draw the query image, t1, t2, and t3 images in a row, use 4-item boxes
            #convert q.original_image to a PIL image
            q = Image.fromarray(q.original_image)
            t1 = Image.fromarray(t1.original_image)
            t2 = Image.fromarray(t2.original_image)
            t3 = Image.fromarray(t3.original_image)

            #draw the images in two columns, 20 rows each
            if i < len(ppms) / 2:
                image.paste(q , (0, 110 * i + 40))
                image.paste(t1, (100, 110 * i + 40 ))
                image.paste(t2, (200, 110 * i + 40 ))
                image.paste(t3, (300, 110 * i + 40 ))
            else:
                image.paste(q , (450, 110 * (i%20) + 40))
                image.paste(t1, (550, 110 * (i%20) + 40 ))
                image.paste(t2, (650, 110 * (i%20) + 40 ))
                image.paste(t3, (750, 110 * (i%20 )+ 40 ))
            
            #underneath each image, draw the individual scores
            draw = ImageDraw.Draw(image)

            if i < len(ppms) / 2:
                #label each image with the query image number underneath the image
                # use a larger font size
                draw.text((0, 110 * i + 100), f'q{i + 1} = {rowscore}', font=font)
                draw.text((100, 110 * i + 100), f't{row[0]} = {t1_score}', font=font)
                draw.text((200, 110 * i + 100), f't{row[1]} = {t2_score}', font=font)
                draw.text((300, 110 * i + 100), f't{row[2]} = {t3_score}', font=font)
            else:
                draw.text((450, 110 * (i%20) + 100), f'q{i + 1} = {rowscore}', font=font)
                draw.text((550, 110 * (i%20) + 100), f't{row[0]} = {t1_score}', font=font)
                draw.text((650, 110 * (i%20) + 100), f't{row[1]} = {t2_score}', font=font)
                draw.text((750, 110 * (i%20) + 100), f't{row[2]} = {t3_score}', font=font)

        # show the image currently being worked on
        #image.show()
        # save the image using the name of the scoring function
        image.save(f'{score_function.__name__}.png')
        # annotate the bottom with the total score and the personal score

    visualize_images(ppms, crowd_matrix, personal_matrix, score_images_color)
    visualize_images(ppms, crowd_matrix, personal_matrix, score_images_texture)
    visualize_images(ppms, crowd_matrix, personal_matrix, score_images_shape)
    visualize_images(ppms, crowd_matrix, personal_matrix, score_images_symmetry)
    visualize_images(ppms, crowd_matrix, personal_matrix, score_images_gestalt)

if __name__ == '__main__':
    main()