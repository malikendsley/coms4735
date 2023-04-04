
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
    print_calibration(buildings)
    
    
def print_stats(buildings: dict):
    # iterate through all objects in the dictionary
    for key in buildings:
        print("=========================================")
        # print the stats for each building
        # round everything to 2 decimal places
        print("Building", id_2_name(key), "has the following stats:")
        print(f'MBR Top Left Coordinates: {buildings[key].MBR[0]:.2f}, {buildings[key].MBR[1]:.2f}')
        print(f'MBR Bottom Right Coordinates: {buildings[key].MBR[2]:.2f}, {buildings[key].MBR[3]:.2f}')
        print(f'Center of Mass Coordinates: {buildings[key].COM[0]:.2f}, {buildings[key].COM[1]:.2f}')
        print(f'Pixel Area: {buildings[key].area}')
        print(f'Diagonal Length: {buildings[key].diag:.2f}')
        print("Intersections:", end=" ")
        building_intersections = intersections(buildings[key], buildings.values())
        # print each building_intersection.name
        if len(building_intersections) == 0:
            print("None")
        else:
            print(*[building.name for building in building_intersections], sep=", ")


def print_calibration(buildings: dict):
    calibration_data = calibrate_what(buildings)
    print("=========================================")
    print("===========Calibration Data==============")
    #print the calibration data for all 3 whats
    #they are inside calibration_data object under size, aspect ratio, and geometry
    print("=========================================")
    print("------------ Size -------------------")
    size_data = calibration_data["size"]
    #print each category next to its interval
    for category in size_data:
        print(category, size_data[category])
    print("------------ Aspect Ratio ---------------")
    aspect_data = calibration_data["aspect"]
    for category in aspect_data:
        print(category, aspect_data[category])
if __name__ == "__main__":
    main()