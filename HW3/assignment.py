
import cv2 as cv
from building import Building, id_2_name, name_2_id
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
    
    calibration_data = calibrate_what(buildings)
    
    master = cv.imread("Labeled.pgm", -1)
    # cv2.imshow("Master", master)
    for key in buildings:
        building = buildings[key]
        # to each building attach a tuple of the 3 what properties
        building.traits = (decide_size(building, calibration_data), decide_aspect_ratio(building, calibration_data), decide_shape(building))
    for key in buildings:
        print(f'"What" properties for {buildings[key].name}:')
        print(f'Size: {buildings[key].traits[0]}')
        print(f'Aspect Ratio: {buildings[key].traits[1]}')
        print(f'Shape: {buildings[key].traits[2]}')
        # get a list of all the other buildings with the same trait tuple
        confused = []
        for key2 in buildings:
            if key != key2 and buildings[key].traits == buildings[key2].traits:
                confused.append(buildings[key2].name)
        print(f'Confusion: {confused if len(confused) > 0 else "None"}')
        print(f'')
   
############################################### #
#                                               #
#              Extra Functions                  #
#                                               #
############################################### #
    
    
def print_stats(buildings: dict):
    # iterate through all objects in the dictionary
    for key in buildings:
        print("=========================================")
        # print the stats for each building
        # round everything to 2 decimal places
        print("Building", id_2_name(key), "has the following stats:")
        print(f"Raw MBR Coordinates: {buildings[key].MBR}")
        print(f'MBR Top Left Coordinates: {buildings[key].MBR[0]:.2f}, {buildings[key].MBR[1]:.2f}')
        print(f'MBR Bottom Right Coordinates: {buildings[key].MBR[2]:.2f}, {buildings[key].MBR[3]:.2f}')
        print(f'Center of Mass Coordinates: {buildings[key].COM[0]:.2f}, {buildings[key].COM[1]:.2f}')
        print(f'Pixel Area: {buildings[key].area}')
        print(f'Diagonal Length: {buildings[key].diag:.2f}')
        print(f'Occupied Ratio: {buildings[key].occupied_ratio:.2f}')
        print("Intersections:", end=" ")
        print(f"Aspect Ratio: {buildings[key].aspect:.2f}")

        # cv.circle(image, (int(buildings[key].MBR[0]), int(buildings[key].MBR[1])), 5, (255, 0, 0), -2)
        # cv.circle(image, (int(buildings[key].MBR[2]), int(buildings[key].MBR[3])), 5, (0, 0, 255), -2)
        # cv.imshow("MBR", image)
        # cv.waitKey(0)
        building_intersections = intersections(buildings[key], buildings.values())
        # print each building_intersection.name
        if len(building_intersections) == 0:
            print("None\n")
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