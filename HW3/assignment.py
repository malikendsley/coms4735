
import cv2 as cv
from building import Building
from util import *

def main():
    # read the image
    img = cv.imread("Labeled.pgm", -1)
    buildings = {}
    
    # for each pixel in the image, check which building it belongs 
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # if we haven't seen this building before, create a new building object, ignore 0
            if img[y, x] != 0 and img[y, x] not in buildings:
                buildings[img[y, x]] = Building(img[y, x], img)
    
    print_stats(buildings)
    
def print_stats(buildings: dict):
    # iterate through all objects in the dictionary
    for key in buildings:
        print("=========================================")
        # print the stats for each building
        print("Building", id_2_name(key), "has the following stats:")
        print("MBR Top Left Coordinates", buildings[key].MBR[0], buildings[key].MBR[1])
        print("MBR Bottom Right Coordinates", buildings[key].MBR[2], buildings[key].MBR[3])
        print("COM Coordinates:", buildings[key].COM)
        print("Pixel Area:", buildings[key].area)
        print("MBR Diagonal Length:", buildings[key].diag)
        print("Intersections:", end=" ")
        building_intersections = intersections(buildings[key], buildings.values())
        # print each building_intersection.name
        if len(building_intersections) == 0:
            print("None")
        else:
            print(*[building.name for building in building_intersections], sep=", ")

    
if __name__ == "__main__":
    main()