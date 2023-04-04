
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
        self.diag = 0.0
        self.MBR, self.COM, self.area, self.diag = self.calc_stats(img)
        MBR_height = self.MBR[1] - self.MBR[3]
        MBR_width = self.MBR[2] - self.MBR[0]
        # divide the smaller dimension by the larger dimension to get the aspect ratio
        if MBR_height > MBR_width:
            self.aspect = MBR_height / MBR_width
        else:
            self.aspect = MBR_width / MBR_height

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
                    #MBR 0 = x1, 1 = y1, 2 = x2, 3 = y2
                    if MBR1x == 0.0 and MBR1y == 0.0:
                        MBR1x, MBR1y = x, y
                    if MBR2x == 0.0 and MBR2y == 0.0:
                        MBR2x, MBR2y = x, y
                    if x < MBR1x:
                        MBR1x = x
                    if y > MBR1y:
                        MBR1y = y
                    if x > MBR2x:
                        MBR2x = x
                    if y < MBR2y:
                        MBR2y = y

                    # update COM
                    COMx += x
                    COMy += y
        COMx /= area
        COMy /= area
        diag = ((MBR2x - MBR1x)**2 + (MBR2y - MBR1y)**2)**0.5
        return (MBR1x, MBR1y, MBR2x, MBR2y), (COMx, COMy), area, diag