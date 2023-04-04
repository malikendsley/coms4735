#read the Table.txt file to generate the lookup table for integer to string conversion

#dict named lookup
table = {}
r_table = {}
#open Table.txt file
#each line in the file is read and split into two parts
#the first part is a number and the second part is a string
#the number is converted to an integer and the string is stored in the lookup table
#store it both ways so that we can convert either way
with open('Table.txt') as f:
    for line in f:
        (key, val) = line.split()
        table[int(key)] = val
        r_table[val] = int(key)

#return the string associated with the integer
def id_2_name(x: int) -> str:
    return table[x]

#return the integer associated with the string
def name_2_id(x: str) -> int:
    return r_table[x]

#return a list of buildings whose MBR intersects with the MBR of the building passed in
def intersections(building, list):
    intersections = []
    for b in list:
        if b.id == building.id:
            continue
        if b.area == 0:
            continue
        #MBR 0 = x1, 1 = y1, 2 = x2, 3 = y2
        #MBR is upper left and lower right
        if building.MBR[0] >= b.MBR[2] or building.MBR[2] <= b.MBR[0] or building.MBR[1] <= b.MBR[3] or building.MBR[3] >= b.MBR[1]:
            continue
        else:
            intersections.append(b)
    return intersections

# given a list of buildings, calibrate the size, aspect ratio, and shape descriptors
def calibrate_what(buildings:dict):
    #make dict into list
    buildings_list = list(buildings.values())
    calibration_data = {}
    # calculate the average size of all buildings
    average = 0
    for b in buildings_list:
        average += b.area
    average /= len(buildings_list)
    smallest = sorted(buildings_list, key=lambda b: b.area)[0].area
    largest = sorted(buildings_list, key=lambda b: b.area)[-1].area

    # the smallest building defined by the smallest building, but also buildings within 10% of the smallest building
    # since they are likely to be the confused with the smallest building
    # the largest and smallest groups are based on the largest and smallest buildings, then padded by 10%
    # the middle group is based on the average size of the buildings, then padded by 20%
    # the other two groups are based on the region left over after the first three groups are calculated
    smallest *= 1.1
    largest *= 0.9

    medium_lower_cutoff = average * 0.8
    medium_upper_cutoff = average * 1.2

    # ensure the intervals do not overlap
    if smallest > medium_lower_cutoff:
        print("Error: smallest and medium_lower_cutoff overlap")
        return  
    if medium_lower_cutoff > medium_upper_cutoff:
        print("Error: medium_lower_cutoff and medium_upper_cutoff overlap")
        return
    if medium_upper_cutoff > largest:
        print("Error: medium_upper_cutoff and largest overlap")
        return
    
    # split the buildings into 5 groups based on size
    calibration_data["size"] = {}
    calibration_data["size"]["smallest"] = (0, smallest)
    calibration_data["size"]["small"] = (smallest + 1, medium_lower_cutoff)
    calibration_data["size"]["medium"] = (medium_lower_cutoff + 1, medium_upper_cutoff)
    calibration_data["size"]["large"] = (medium_upper_cutoff + 1, largest)
    calibration_data["size"]["largest"] = (largest, 1000000)
    
    
    
    #aspect is given by ratio of MBR width to MBR height, ranging from 0 to 1
    # the categories are "narrow", "medium-wide", and "wide"
    # narrow is near a value of 0, medium-wide is near a value of 0.5, and wide is near a value of 1
    average_aspect = 0
    
    for b in buildings_list:
        average_aspect += b.aspect
        
    average_aspect /= len(buildings_list)
    # print("average aspect", average_aspect)
    # min_aspect = sorted(buildings_list, key=lambda b: (b.MBR[2] - b.MBR[0]) / (b.MBR[1] - b.MBR[3]))[0].area
    # max_aspect = sorted(buildings_list, key=lambda b: (b.MBR[2] - b.MBR[0]) / (b.MBR[1] - b.MBR[3]))[-1].area
    
    calibration_data["aspect"] = {}
    calibration_data["aspect"]["narrow"] = (0, average_aspect * 0.8)
    calibration_data["aspect"]["medium-wide"] = (average_aspect * 0.8, average_aspect * 1.2)
    calibration_data["aspect"]["wide"] = (average_aspect * 1.2, 1)
    
    return calibration_data

    # decide the shape of the building from among the categories
    # square, rectangular, I-shaped, C-shaped, L-shaped, asymmetric
    # square is a building with an aspect ratio near 1, and whose bounding box leaves few gaps
    # rectangular is a building with an aspect ratio less than .8 and the bounding box leaves few gaps 
    # if the bounding box leaves gaps, then check for the below shapes
    # an L shaped building becomes rectangular when folded along the diagonal
    # a C shaped building becomes L shaped when folded in half
    # an I shaped building becomes C shaped when folded in half parallel to the longer side
    # asymmetric is a building that does not fit into any of the other categories
    # so, test for square, then rectangular, then L, then C, then I so that the most specific shape is chosen
    def decide_shape(building):
        pass
    def isSquare(building):
        pass
    def isRectangular(building):
        pass
    def isLShaped(building):
        pass
    def isCShaped(building):
        pass
    def isIShaped(building):
        pass
    