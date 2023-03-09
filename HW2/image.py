from typing import Tuple
import cv2
import numpy as np

laplace_matrix = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# class for handling images, precomputes a lot of useful information
class PPMImage:
    # takes a filename, and all of the hyper parameters (includes defaults)
    def __init__(self, filename: str, color_depth = (8,8,8), laplace_depth = 11, binary_threshold = 128, binary_symmetry_threshold = 128):
        # basic information and the actual image
        self.filename = filename
        self.bit_depths = color_depth
        self.laplace_depth = laplace_depth
        self.original_image = cv2.imread(filename)
        
        # create versions of image
        self.color = self.init_color(self.original_image, color_depth)
        self.grayscale = self.init_grayscale(self.original_image)
        self.texture = self.init_texture(self.grayscale, laplace_depth)
        self.binary_for_shape = self.init_binary(self.grayscale, binary_threshold)
        self.binary_for_symmetry = self.init_binary(self.grayscale, binary_symmetry_threshold)
        
        # make the color histogram for the image
        bins = [np.linspace(0, 2**color_depth[i], 2**color_depth[i] + 1) for i in range(3)]
        self.color_histogram, _ =  np.histogramdd(self.color.reshape(-1, 3), bins=bins)
    
        # make the texture histogram for the image
        bins = np.linspace(0, 2**laplace_depth, 2**laplace_depth + 1)
        self.texture_histogram, _ = np.histogram(self.texture.reshape(-1), bins=bins)
    
    #############################
    # Init and Helper Functions #
    #############################
    
    # change the bit depth of the image without reloading it
    def set_color_depth(self, bit_depths):
        self.bit_depths = bit_depths
        self.color = self.init_color(self.original_image, bit_depths)
        bins = [np.linspace(0, 2**bit_depths[i], 2**bit_depths[i] + 1) for i in range(3)]
        self.color_histogram, _ =  np.histogramdd(self.color.reshape(-1, 3), bins=bins)
    
    # change the bit depth of the texture without reloading it
    def set_laplace_depth(self, laplace_depth):
        self.laplace_depth = laplace_depth
        self.texture = self.init_texture(self.grayscale, laplace_depth)
        bins = np.linspace(0, 2**laplace_depth, 2**laplace_depth + 1)
        self.texture_histogram, _ = np.histogram(self.texture.reshape(-1), bins=bins)
    
    # change the binary threshold without reloading it
    def set_binary_threshold(self, threshold):
        self.binary_for_shape = self.init_binary(self.grayscale, threshold)
    
    # change the binary threshold for symmetry without reloading it
    def set_binary_symmetry_threshold(self, threshold):
        self.binary_for_symmetry = self.init_binary(self.grayscale, threshold)
    
    # create a color image with the given bit depth of the original image
    def init_color(self, image: cv2.Mat, color_depth=(8,8,8)):
        color = image.copy()
        for i in range(3):
            color[:,:,i] = color[:,:,i] >> (8 - color_depth[i])
        return color
    
    # create a grayscale image from the original image
    def init_grayscale(self, image: cv2.Mat):
        grayscale = image.copy()
        grayscale = np.sum(grayscale, axis=2) // 3
        grayscale = grayscale.astype(np.uint8)
        return grayscale
    
    # create a texture image from the grayscale image
    def init_texture(self, grayscale_image: cv2.Mat, laplace_depth=11):
        texture = grayscale_image.copy().astype(np.int16)
        texture = np.abs(cv2.filter2D(texture, -1, laplace_matrix))
        # you originally need 11 bits to represent the max value of 2048
        # reduce the bit depth to make the histogram more useful
        texture  = texture >> (11 - laplace_depth)
        return texture
    
    # create a binary image from the grayscale image
    def init_binary(self, image: cv2.Mat, threshold=128) -> np.ndarray:
        return (image >= threshold).astype(np.uint8)
    
    # # # # # # # # # # #
    # Display Functions #
    # # # # # # # # # # #
    
    # show the full image
    def show_orig(self, factor=1):
        pic = cv2.resize(self.original_image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # show the bit reduced image
    def show_color(self, factor=1):
        pic = self.color.copy()
        #scale the values back to 255
        for i in range(3):
            pic[:,:,i] = pic[:,:,i] * (255 // (2**self.bit_depths[i] - 1))
        pic = cv2.resize(pic, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # show the grayscale image
    def show_gray(self, factor=1):
        pic = cv2.resize(self.grayscale, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # show the edge image
    def show_texture(self, factor=1):
        pic = self.texture.copy()
        #normalize the values to 0-255
        pic = pic * (255 // (2**self.laplace_depth - 1))
        pic = cv2.resize(pic, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # show the binary image
    def show_shape_binary(self, factor=1):
        pic = cv2.resize(self.binary_for_shape, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        #convert 1s to 255s
        pic[pic == 1] = 255
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show the binary image for symmetry
    def show_symmetry_binary(self, factor=1):
        pic = cv2.resize(self.binary_for_symmetry, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        #convert 1s to 255s
        pic[pic == 1] = 255
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()