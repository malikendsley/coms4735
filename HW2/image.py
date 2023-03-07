# class for reading ppm image into numpy 3d array
import cv2
import numpy as np


class PPMImage:
    
    #takes a filename, and a 3-tuple of the RGB bit depths for the histogram
    def __init__(self, filename, id, bit_depths=(8,8,8)):
        self.bit_depths = bit_depths
        self.id = id
        self.filename = filename
        self.image = cv2.imread(filename)
        # copy the image to a new var
        self.reduced = self.image.copy()
        
        #for each channel, reduce the bit depth by the specified amount
        for i in range(3):
            self.reduced[:,:,i] = self.reduced[:,:,i] >> (8 - bit_depths[i])
                     
        # for all pixels add up the values of the channels and divide by 3
        self.grayscale = self.image.copy()
        # the data is in format [height][width][channel] so we need to sum over the channel axis and make int
        self.grayscale = np.sum(self.grayscale, axis=2) // 3
        self.grayscale = self.grayscale.astype(np.uint8)

        # make the histogram for the image
        bins = [np.linspace(0, 2**bit_depths[i], 2**bit_depths[i] + 1) for i in range(3)]
        self.histogram, _ =  np.histogramdd(self.reduced.reshape(-1, 3), bins=bins)
        
    def display(self, factor=1):
        pic = cv2.resize(self.image, (0,0), fx=factor, fy=factor)
        cv2.imshow(str(self.id), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def display_reduced(self, factor=1):
        #avoid modifying the original image
        pic = self.reduced.copy()
        # blow up the data to 255 so we can display it
        for i in range(3):
            pic[:,:,i] = pic[:,:,i] * (255 // (2**self.bit_depths[i] - 1))
        pic = cv2.resize(pic, (0,0), fx=factor, fy=factor)
        cv2.imshow(str(self.id), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_gray(self, factor=1):
        # image is stored as [height][width] = intensity
        # display it as a grayscale image
        pic = cv2.resize(self.grayscale, (0,0), fx=factor, fy=factor)
        cv2.imshow(str(self.id), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #change the bit depth of the image without reloading it
    def set_bit_depths(self, bit_depths):
        self.bit_depths = bit_depths
        self.reduced = self.image.copy()
        for i in range(3):
            self.reduced[:,:,i] = self.reduced[:,:,i] >> (8 - bit_depths[i])
        bins = [np.linspace(0, 2**bit_depths[i], 2**bit_depths[i] + 1) for i in range(3)]
        self.histogram, _ =  np.histogramdd(self.reduced.reshape(-1, 3), bins=bins)
    
