# class for reading ppm image into numpy 3d array
from typing import Tuple
import cv2
import numpy as np

laplace_matrix = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

class PPMImage:
    
    #takes a filename, and a 3-tuple of the RGB bit depths for the histogram
    def __init__(self, filename: str, color_depth: Tuple[int, int, int] = (8,8,8), laplace_depth = 8):
        self.bit_depths = color_depth
        self.laplace_depth = laplace_depth
        self.filename = filename
        self.original_image = cv2.imread(filename)
        
        # create versions of image
        self.color = self.init_color(self.original_image, color_depth)
        self.grayscale = self.init_grayscale(self.original_image)
        self.texture = self.init_texture(self.grayscale, laplace_depth)
        
        # make the color histogram for the image
        bins = [np.linspace(0, 2**color_depth[i], 2**color_depth[i] + 1) for i in range(3)]
        self.color_histogram, _ =  np.histogramdd(self.color.reshape(-1, 3), bins=bins)
        
        #make the texture histogram for the image
        self.texture_histogram, _ = np.histogramdd(self.texture.reshape(-1, 1), bins=[np.linspace(-255, 255, 511)])
         
    #change the bit depth of the image without reloading it
    def set_color_depth(self, bit_depths):
        self.bit_depths = bit_depths
        self.color = self.init_color(self.original_image, bit_depths)
        bins = [np.linspace(0, 2**bit_depths[i], 2**bit_depths[i] + 1) for i in range(3)]
        self.color_histogram, _ =  np.histogramdd(self.color.reshape(-1, 3), bins=bins)
    
    def set_laplace_depth(self, laplace_depth):
        self.laplace_depth = laplace_depth
        self.texture = self.init_texture(self.grayscale, laplace_depth)
        self.texture_histogram, _ = np.histogramdd(self.texture.reshape(-1, 1), bins=[np.linspace(-255, 255, 511)])
    
    def init_color(self, image: cv2.Mat, color_depth=(8,8,8)):
        color = image.copy()
        for i in range(3):
            color[:,:,i] = color[:,:,i] >> (8 - color_depth[i])
        return color
    
    def init_grayscale(self, image: cv2.Mat):
        grayscale = image.copy()
        grayscale = np.sum(grayscale, axis=2) // 3
        grayscale = grayscale.astype(np.uint8)
        return grayscale
    
    def init_texture(self, grayscale_image: cv2.Mat, laplace_depth=8):
        texture = grayscale_image.copy()
        texture = cv2.filter2D(texture, -1, laplace_matrix)
        texture  = texture >> (8 - laplace_depth)
        return texture
    
    def laplace_convolve(self, image):
        padded = np.pad(image, 1, mode='edge')
        convolved = np.zeros(image.shape)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                convolved[i,j] = np.sum(padded[i:i+3, j:j+3] * laplace_matrix)
        print("Convolved\n", convolved)
    
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
        #avoid modifying the original image
        pic = self.color.copy()
        # blow up the data to 255 so we can display it
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
        #scale the values to 255 with multiplication
        pic = pic * (255 // (2**self.laplace_depth - 1))
        pic = cv2.resize(pic, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(str(self.filename), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
