from image import PPMImage
import numpy as np
import math
import itertools

######################
# Distance Functions #
######################

# get the L1 distance between two images w.r.t color
def distance_color(img1: PPMImage, img2: PPMImage):
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.color_histogram - img2.color_histogram)) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)

# get the L1 distance between two images w.r.t texture
def distance_texture(img1: PPMImage, img2: PPMImage):
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    return np.sum(np.abs(img1.texture_histogram - img2.texture_histogram)) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)

# get the L1 distance between two images w.r.t shape
def distance_shape(img1: PPMImage, img2: PPMImage):
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    #xor causes 1 to be returned if the pixels are different, so summing the result gives the number of different pixels
    #then you just have to normalize by the number of pixels compared
    return np.sum(np.logical_xor(img1.binary_for_shape, img2.binary_for_shape)).astype(np.float32) / (img1.original_image.shape[0] * img1.original_image.shape[1] * 2.0)

#get the symmetry score of an image
def distance_symmetry(img: PPMImage):
    #if odd number of columns, ignore the middle column
    matrix = img.binary_for_symmetry.copy()
    if matrix.shape[1] % 2 == 1:
        matrix = np.delete(matrix, matrix.shape[1] // 2, 1)
    columns = matrix.shape[1] // 2
    #sum the number of times the pixels are different
    distance = np.sum(np.logical_xor(matrix[:, :columns], np.fliplr(matrix[:, columns:]))).astype(np.float32)
    #normalize by the number of pixels in one half of the image
    distance /= (matrix.shape[0] * columns)
    return distance

#get the gestalt distance between two images 
#uses a linear combination of the color, texture, shape, and symmetry distances with the given weights
def distance_gestalt(img1: PPMImage, img2: PPMImage, weights: "list[float]"):
    if img1.bit_depths != img2.bit_depths:
        raise ValueError("Images must have same bit depth")
    if img1.original_image.shape != img2.original_image.shape:
        raise ValueError("Images must have same dimensions")

    distances = np.empty(4)
    distances[0] = (distance_color(img1, img2))
    distances[1] = (distance_texture(img1, img2))
    distances[2] = (distance_shape(img1, img2))
    distances[3] = (np.abs(distance_symmetry(img1) - distance_symmetry(img2)))
    
    #multiply each weight by the corresponding distance
    distances = np.dot(distances, weights)
    
    return np.sum(distances)
    
#####################
# Scores and Tuners #    
#####################

# score the images according to color against the crowd matrix
def score_images_color(ppms, crowd_matrix, personal_matrix=None):
    # measure the distance between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = distance_color(ppms[i], ppms[j])
    
    # get the 3 closest images for each image, then look up the crowd score for each of those
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    
    # get the personal score for the system
    personal_score = None
    if personal_matrix is not None:
        personal_score = row_intersections(personal_matrix, top3 + 1)
    
    return np.sum(crowd3), top3 + 1, personal_score

# optimizes the color bit depths for the given images to give the best score
def color_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    best = -math.inf
    best_bit_depths = (1, 1, 1)
    # iterate from 111 to 666, going to 888 is too slow and known to be worse
    for r in range(1, 7):
        for g in range(1, 7):
            for b in range(1, 7):
                for ppm in ppms:
                    ppm.set_color_depth((r, g, b))
                score = score_images_color(ppms, crowd_matrix)[0]
                if score > best:
                    best = score
                    best_bit_depths = (r, g, b)
                    print(f'new best: {best_bit_depths} with score {best}')
    print(f'Best color bit depths: {best_bit_depths}')
    return best_bit_depths

# score the images according to texture against the crowd matrix
def score_images_texture(ppms, crowd_matrix, personal_matrix=None):
    # measure the distance between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = distance_texture(ppms[i], ppms[j])
    
    # get the 3 closest images for each image, then look up the crowd score for each of those
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    
    # get the personal score for the system
    personal_score = None
    if personal_matrix is not None:
        personal_score = row_intersections(personal_matrix, top3 + 1)
    
    return np.sum(crowd3), top3 + 1, personal_score

# optimizes the texture bit depth for the given images to give the best score
def texture_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    best = -math.inf
    best_bit_depth = 1
    # iterate from 1-11 bits of binning
    for d in range(1, 12):
        for ppm in ppms:
            ppm.set_laplace_depth(d)
        score = score_images_texture(ppms, crowd_matrix)[0]
        if score > best:
            best = score
            best_bit_depth = d
            print(f'new best: {best_bit_depth} with score {best}')
    print(f'Best texture bit depths: {best_bit_depth}')
    return best_bit_depth

# score the images according to shape against the crowd matrix
def score_images_shape(ppms, crowd_matrix, personal_matrix=None):
    # measure the distance between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = distance_shape(ppms[i], ppms[j])

    # get the 3 closest images for each image, then look up the crowd score for each of those
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    
    # get the personal score for the system
    personal_score = None
    if personal_matrix is not None:
        personal_score = row_intersections(personal_matrix, top3 + 1)
    
    return np.sum(crowd3), top3 + 1, personal_score

# optimizes the shape bit depth for the given images to give the best score
def shape_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    best = -math.inf
    threshold = 1
    # iterate from threshold of 1 to 255
    for t in range(1, 256):
        for ppm in ppms:
            ppm.set_binary_threshold(t)
        score = score_images_shape(ppms, crowd_matrix)[0]
        if score > best:
            best = score
            threshold = t
            print(f'new best: {threshold} with score {best}')
    print(f'Best shape bit depths: {threshold}')
    return threshold

# score the images according to symmetry against the crowd matrix
def score_images_symmetry(ppms, crowd_matrix, personal_matrix=None):
    # get the symmetry score for each image
    scores = np.zeros(len(ppms))
    for i in range(len(ppms)):
        scores[i] = distance_symmetry(ppms[i])
    
    top3 = np.zeros((len(ppms), 3), dtype=int)
    for i, score in enumerate(scores):
        #find the 3 closest image indexes by finding the smallest delta
        top3[i] = np.argsort(np.abs(scores - score))[1:4]
    # use these to look up the crowd scores
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    
    # get the personal score for the system
    personal_score = None
    if personal_matrix is not None:
        personal_score = row_intersections(personal_matrix, top3 + 1)
    
    return np.sum(crowd3), top3 + 1, personal_score

# optimizes the symmetry bit depth for the given images to give the best score
def symmetry_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]"):
    best = -math.inf
    threshold = 1
    # iterate from threshold of 1 to 255
    for t in range(1, 256):
        for ppm in ppms:
            ppm.set_binary_symmetry_threshold(t)
        score = score_images_symmetry(ppms, crowd_matrix)[0]
        if score > best:
            best = score
            threshold = t
            print(f'new best: {threshold} with score {best}')
    print(f'Best symmetry bit depths: {threshold}')
    return threshold

# score the images according to gestalt against the crowd matrix
def score_images_gestalt(ppms, crowd_matrix, weights, personal_matrix=None):
    # measure the distance between each image
    distance_matrix = np.zeros((len(ppms), len(ppms)))
    for i in range(len(ppms)):
        for j in range(len(ppms)):
            distance_matrix[i][j] = distance_gestalt(ppms[i], ppms[j], weights)
    
    # get the 3 closest images for each image, then look up the crowd score for each of those
    top3 = np.array([np.argsort(distance_matrix[i])[1:4] for i in range(len(ppms))])
    crowd3 = np.array([crowd_matrix[i][top3[i]] for i in range(len(ppms))])
    
    # get the personal score for the system
    personal_score = None
    if personal_matrix is not None:
        personal_score = row_intersections(personal_matrix, top3 + 1)
    
    return np.sum(crowd3), top3 + 1, personal_score

# optimizes the gestalt bit depth for the given images to give the best score
def gestalt_hyper_tuner(crowd_matrix, ppms: "list[PPMImage]", resolution=0.1):
    best = -math.inf
    best_weights = np.zeros(4)
    iteration = 0

    #create a list of weight values to test
    weights = [i*resolution for i in range(int((1 / resolution) + 1))]
    # generate all possible combinations
    combinations = itertools.product(weights, repeat=4)
    # filter out combinations that don't add up to 1
    valid_combinations = [c for c in combinations if sum(c) == 1]
    # round to match the resolution to avoid floating point errors, a bit hacky but it works
    valid_combinations = [[round(v, len(str(resolution))-2) for v in c] for c in valid_combinations]

    best_weights = valid_combinations[0]
    # this takes a while, so more print statements are added to show progress
    print("Starting hyperparameter tuning...")
    for combo in valid_combinations:
        iteration += 1
        if iteration % 50 == 0:
            print(f'Iteration {iteration}, testing weights {combo}, best score so far is {best}')
        score = score_images_gestalt(ppms, crowd_matrix, combo)[0]
        if score > best:
            best = score
            best_weights = combo
            print(f'new best: {combo} with score {best}')
    print(f'Best weights: {best_weights}')
    return best_weights

##################
# Misc Functions #
##################

# for each row, find the number of agreements between the top 3 images and the personal matrix
def row_intersections(top3, personal_matrix):
    personal_sets = [set(row) for row in personal_matrix.astype(int)]
    matches = [len(personal_sets[i].intersection(row)) for i, row in enumerate(top3)]
    
    return np.sum(matches)