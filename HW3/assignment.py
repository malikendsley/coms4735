
import cv2 as cv
from building import Building, id_2_name, name_2_id
from util import *

def main():
    # read the image
    img = cv.imread("Labeled.pgm", -1)
    buildings = {}
    
    # for each pixel in the image, check which building it belongs to
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # if we haven't seen this building before, create a new building object, ignore 0
            if img[y, x] != 0 and img[y, x] not in buildings:
                buildings[img[y, x]] = Building(img[y, x], img)
    
    ################################################
                    #STEP 1#
    ################################################
    
    
    # print_stats(buildings)
    #vprint_calibration(buildings)
    
    # these are helper functions to calculate the thresholds for the various steps
    what_data = calibrate_what(buildings)
    where_data = calibrate_where(buildings)
    
    ################################################
                    #STEP 2#
    ################################################
    
    
    # master = cv.imread("Labeled.pgm", -1)
    # cv2.imshow("Master", master)
    # for key in buildings:
    #     building = buildings[key]
    #     # to each building attach a tuple of the 3 what properties
    #     building.traits = (decide_size(building, what_data), decide_aspect_ratio(building, what_data), decide_shape(building))
    # for key in buildings:
    #     print("==================================================================")
    #     print(f'"What" properties for {buildings[key].name}:')
    #     print(f'Size: {buildings[key].traits[0]}')
    #     print(f'Aspect Ratio: {buildings[key].traits[1]}')
    #     print(f'Shape: {buildings[key].traits[2]}')
    #     # get a list of all the other buildings with the same trait tuple
    #     confused = []
    #     for key2 in buildings:
    #         if key != key2 and buildings[key].traits == buildings[key2].traits:
    #             confused.append(buildings[key2].name)
    #     print(f'Confusion: {confused if len(confused) > 0 else "None"}')
    #     print(f'Minimization: ')
    
    ################################################
                    #STEP 3#
    ################################################
    
    # for key in buildings:
    #     building = buildings[key]
    #     building.wheretraits = (decide_vertical(building, where_data), decide_horizontal(building, where_data), decide_orientation(building))
    
    # for key in buildings:
    #     print("==================================================================")
    #     print(f"Where properties for {buildings[key].name}:")
    #     print(f'Vertical Position: {buildings[key].wheretraits[0]}')
    #     print(f'Horizontal Position: {buildings[key].wheretraits[1]}')
    #     print(f'Orientation: {buildings[key].wheretraits[2]}')
    #     confused = []
    #     for key2 in buildings:
    #         if key != key2 and buildings[key].wheretraits == buildings[key2].wheretraits:
    #             confused.append(buildings[key2].name)
    #     print(f'Confusion: {confused if len(confused) > 0 else "None"}')
    #     print(f'Minimization: ')
        
    ################################################
                    #STEP 4#
    ################################################

    # create dictionaries to store source-target and target-source mappings
    source_to_targets = {}
    target_to_sources = {}

    # iterate through all buildings
    for source_key in buildings:
        # list all target buildings T that are nearTo(S, T)
        targets = []
        for target_key in buildings:
            if source_key != target_key and nearTo(buildings[source_key], buildings[target_key], tuning=0.025):
                targets.append(buildings[target_key].name)
                # add target to the target-source mapping
                if target_key not in target_to_sources:
                    target_to_sources[target_key] = []
                target_to_sources[target_key].append(buildings[source_key].name)
        # add targets to the source-target mapping
        if source_key not in source_to_targets:
            source_to_targets[source_key] = []
        source_to_targets[source_key] = targets

    print("Source to Target Mapping ===============")
    # print the source-target mapping
    for key in source_to_targets:
        print(f"{buildings[key].name} is near to: {', '.join(source_to_targets[key])}")
    print("Target to Source Mapping ===============")
    # print the target-source mapping
    for key in target_to_sources:
        print(f"These buildings are near to {buildings[key].name}: {', '.join(target_to_sources[key])}")

    # Confusion Analysis
    print("Confusion Analysis ===============")
    # Find source with most targets and source with least targets
    max_targets = 0
    min_targets = float('inf')
    max_source = None
    min_source = None
    for key in source_to_targets:
        num_targets = len(source_to_targets[key])
        if num_targets > max_targets:
            max_targets = num_targets
            max_source = buildings[key].name
        if num_targets < min_targets:
            min_targets = num_targets
            min_source = buildings[key].name
    print(f"Most confused source building: {max_source} (with {max_targets} targets)")
    print(f"Least confused source building: {min_source} (with {min_targets} targets)")

    # Find target with most sources and target with least sources
    max_sources = 0
    min_sources = float('inf')
    max_target = None
    min_target = None
    for key in target_to_sources:
        num_sources = len(target_to_sources[key])
        if num_sources > max_sources:
            max_sources = num_sources
            max_target = buildings[key].name
        if num_sources < min_sources:
            min_sources = num_sources
            min_target = buildings[key].name
    print(f"Most confused target building: {max_target} (with {max_sources} sources)")
    print(f"Least confused target building: {min_target} (with {min_sources} sources)")

    # create a dictionary to store each building's potential source buildings
    potential_sources = {}

    # iterate through all buildings
    for target_key in buildings:
        potential_sources[target_key] = []
        # iterate through all other buildings to find potential source buildings
        for source_key in buildings:
            if source_key != target_key and nearTo(buildings[source_key], buildings[target_key], tuning=0.03):
                potential_sources[target_key].append(source_key)

    # create a dictionary to store each building's landmark source building
    landmark_sources = {}

    # iterate through all buildings
    for target_key in buildings:
        # initialize a dictionary to count how many times each potential source building is used as a source for other target buildings
        source_counts = {}
        for source_key in potential_sources[target_key]:
            source_counts[source_key] = 0
        # iterate through all other target buildings to count the potential source buildings used as sources
        for other_key in buildings:
            if other_key != target_key:
                for source_key in potential_sources[other_key]:
                    if source_key in source_counts:
                        source_counts[source_key] += 1
        # find the potential source building with the highest count
        max_count = 0
        landmark_source = None
        for source_key in source_counts:
            if source_counts[source_key] > max_count:
                max_count = source_counts[source_key]
                landmark_source = source_key
        landmark_sources[target_key] = landmark_source

    # print the landmark sources for each building
    print("Landmark Sources ===============")
    for key in landmark_sources:
        if landmark_sources[key] is not None:
            print(f"{buildings[key].name} landmark source: {buildings[landmark_sources[key]].name}")
        else:
            print(f"{buildings[key].name} landmark source: None")
        
    
############################################### #
#                                               #
#              Extra Functions                  #
#                                               #
############################################### #
    
# print the deliverables for step 1
def print_stats(buildings: dict):
    # iterate through all objects in the dictionary
    for key in buildings:
        print("==================================================================")
        # print the stats for each building
        # round everything to 2 decimal places
        print(f"Building Name: {buildings[key].name} --- ID: {key}")
        print(f'MBR:  ({buildings[key].MBR[0]:.2f}, {buildings[key].MBR[1]:.2f}) to ({buildings[key].MBR[2]:.2f}, {buildings[key].MBR[3]:.2f})')
        print(f'Center of Mass: ({buildings[key].COM[0]:.2f}, {buildings[key].COM[1]:.2f}), Pixel Area: {buildings[key].area}')
        print(f'Diagonal Length: {buildings[key].diag:.2f}')
        print("Intersections:", end=" ")

        building_intersections = intersections(buildings[key], buildings.values())
        # print each building_intersection.name
        if len(building_intersections) == 0:
            print("None")
        else:
            print(*[building.name for building in building_intersections], sep=", ")

# print the what calibration data for debugging
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