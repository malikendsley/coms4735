
from typing import Tuple
import util
import cv2 as cv

# buildings are read from the map based on an integer grayscale value
class Building:
    def __init__(self, id: int, img: cv.Mat):
        self.id = id
        self.name = util.id_2_name(id)
        self.MBR = (0.0, 0.0, 0.0, 0.0)
        self.COM = (0.0, 0.0)
        self.area = 0
        self.MBR, self.COM, self.area = self.calc_stats(img)

    def calc_stats(self, img: cv.Mat)-> Tuple[Tuple[float, float, float, float], Tuple[float, float], int]:
        area = 0
        MBR1x, MBR1y, MBR2x, MBR2y = 0.0, 0.0, 0.0, 0.0
        COMx, COMy = 0.0, 0.0
        
        # iterate through all pixels in the image
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] == self.id:
                    area += 1
                    # update MBR1 and MBR2
                    if MBR1x == 0 or x < MBR1x:
                        MBR1x = x
                    if MBR1y == 0 or y < MBR1y:
                        MBR1y = y
                    if MBR2x == 0 or x > MBR2x:
                        MBR2x = x
                    if MBR2y == 0 or y > MBR2y:
                        MBR2y = y

                    # update COM
                    COMx += x
                    COMy += y
        COMx /= area
        COMy /= area
        return (MBR1x, MBR1y, MBR2x, MBR2y), (COMx, COMy), area