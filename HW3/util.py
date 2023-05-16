#read the Table.txt file to generate the lookup table for integer to string conversion

#dict named lookup
import math
import cv2
import numpy as np

table = {}
r_table = {}

# use this in case you need to use the original image
original_image = cv2.imread('Labeled.pgm')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

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
        # find overlapping rectangles
        if building.MBR[0] >= b.MBR[2] or building.MBR[2] <= b.MBR[0] or building.MBR[1] >= b.MBR[3] or building.MBR[3] <= b.MBR[1]:
            continue
        else:
            intersections.append(b)
    return intersections

# given a list of buildings, calibrate the size, aspect ratio, and shape descriptors
def calibrate_what(buildings:dict):
    #make dict into list
    buildings_list = list(buildings.values())
    what_calibration_data = {}
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
    what_calibration_data["size"] = {}
    what_calibration_data["size"]["smallest"] = (0, smallest)
    what_calibration_data["size"]["small"] = (smallest + 1, medium_lower_cutoff)
    what_calibration_data["size"]["medium"] = (medium_lower_cutoff + 1, medium_upper_cutoff)
    what_calibration_data["size"]["large"] = (medium_upper_cutoff + 1, largest)
    what_calibration_data["size"]["largest"] = (largest, 1000000)
    

    #aspect is given by ratio of MBR width to MBR height, ranging from 0 to 1
    # the categories are "narrow", "medium-wide", and "wide"
    # narrow is near a value of 0, medium-wide is near a value of 0.5, and wide is near a value of 1
    average_aspect = 0
    
    for b in buildings_list:
        average_aspect += b.aspect
        
    average_aspect /= len(buildings_list)
    
    what_calibration_data["aspect"] = {}
    what_calibration_data["aspect"]["narrow"] = (0, average_aspect * 0.8)
    what_calibration_data["aspect"]["medium-wide"] = (average_aspect * 0.8, average_aspect * 1.2)
    what_calibration_data["aspect"]["wide"] = (average_aspect * 1.2, 1)
    
    return what_calibration_data

###############################
#                             #
#   Shape Detection Functions #
#                             #
###############################

def decide_shape(building):
    # print(f"===============Deciding shape for building {building.name}===============")
    #cv2.imshow("building", building.img)
    # cv2.waitKey(0)
    if isIShaped(building):
        return "I-shaped"
    elif isSquare(building):
        return "square"
    elif isRectangular(building):
        return "rectangular"
    elif isLShaped(building):
        return "L-shaped"
    elif isCShaped(building):
        return "C-shaped"
    else:
        # print ("Building is asymmetric")
        return "asymmetric"

def isSquare(building):
    #approximate a 4 line bounding box for the building
    if building.aspect > 0.9 and building.occupied_ratio > 0.8:
        # print(f"Buidling is square, aspect ratio {building.aspect}, occupied ratio {building.occupied_ratio}")
        return True
    # print("Building is not square")
    return False

def isRectangular(building):
    if building.occupied_ratio > 0.8:
        # print(f"Buidling is rectangular, occupied ratio {building.occupied_ratio}")
        return True
    # print("Building is not rectangular")
    return False

def isLShaped(building):
    # folding the building along the diagonal should make it rectangular
    # if not, it's a more complex shape
    # begin by aligning the origin to the MBR's top left corner and slicing out the shape
    slice = building.img[building.MBR[1]:building.MBR[3], building.MBR[0]:building.MBR[2]]
    offsetX = building.MBR[0]
    offsetY = building.MBR[1]
    newCOMX, newCOMY = building.COM[0] - offsetX, building.COM[1] - offsetY
    newCOMX = round(newCOMX)
    newCOMY = round(newCOMY)
    foldDirRising = None
    # if the COM is in the top left, fold along the bottom right diagonal
    if newCOMX < slice.shape[1] / 2 and newCOMY < slice.shape[0] / 2:
        foldDirRising = True
    # if the COM is in the top right, fold along the bottom left diagonal
    elif newCOMX > slice.shape[1] / 2 and newCOMY < slice.shape[0] / 2:
        foldDirRising = False
    # if the COM is in the bottom left, fold along the top right diagonal
    elif newCOMX < slice.shape[1] / 2 and newCOMY > slice.shape[0] / 2:
        foldDirRising = False
    # if the COM is in the bottom right, fold along the top left diagonal
    elif newCOMX > slice.shape[1] / 2 and newCOMY > slice.shape[0] / 2:
        foldDirRising = True
    else:
        print("odd...")
    # "fold" the building by checking each pixel with its mirror image (swap x and y)
    flipped = None
    if foldDirRising:
        flipped = np.transpose(slice)
    else:
        flipped = np.flipud(np.transpose(slice))
    
    destination = np.zeros(slice.shape, dtype=np.uint8)
    # for an x y flip to avoid overflow, only flip up to the smaller of the two dimensions
    cap = min(slice.shape[0], slice.shape[1])
    # if either flipped or the original contains a pixel, put it in the destination
    # only check above the diagonal
    for i in range(cap):
        for j in range(i):
            if flipped[i][j] > 0 or slice[i][j] > 0:
                destination[i][j] = 255
    
    # cv2.imshow("building2", destination)
    # if you are left with a rectangle you most likely started with an L-shaped building
    # check the rectangularness by checking the density and the aspect ratio
    MBRx1, MBRy1, MBRx2, MBRy2 = 0, 0, 0, 0
    area = 0
    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            if destination[i][j] > 0:
                area += 1
                if i < MBRy1:
                    MBRy1 = i
                if i > MBRy2:
                    MBRy2 = i
                if j < MBRx1:
                    MBRx1 = j
                if j > MBRx2:
                    MBRx2 = j
    width = MBRx2 - MBRx1 + 1
    height = MBRy2 - MBRy1 + 1
    occupied_ratio = area / (width * height)
    aspect = 0
    if width < height:
        aspect = width / height
    else:
        aspect = height / width
    img = destination.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # if the occupied ratio goes up significantly, then the original building was L-shaped
    if occupied_ratio > building.occupied_ratio and occupied_ratio > .75:
        # print("Building is L-shaped")
        return True
    else:
        # print("Building is not L-shaped")
        return False

def isCShaped(building):
    # repeat the folding process, but twice
    slice = building.img[building.MBR[1]:building.MBR[3], building.MBR[0]:building.MBR[2]]
    offsetX = building.MBR[0]
    offsetY = building.MBR[1]
    newCOMX = round(building.COM[0]) - offsetX
    newCOMY = round(building.COM[1]) - offsetY
    newCOMX = round(newCOMX)
    newCOMY = round(newCOMY)
    foldDirRising = None
    if (newCOMX < slice.shape[1] / 2 and newCOMY < slice.shape[0] / 2) or (newCOMX > slice.shape[1] / 2 and newCOMY > slice.shape[0] / 2):
        foldDirRising = True
    elif (newCOMX > slice.shape[1] / 2 and newCOMY < slice.shape[0] / 2) or (newCOMX < slice.shape[1] / 2 and newCOMY > slice.shape[0] / 2):
        foldDirRising = False
    else:
        print("odd...")
    flipped = None
    if foldDirRising:
        flipped = np.transpose(slice)
    else:
        flipped = np.flipud(np.transpose(slice))
    destination = np.zeros(slice.shape, dtype=np.uint8)
    cap = min(slice.shape[0], slice.shape[1])
    for i in range(cap):
        for j in range(i if foldDirRising else cap - i):
            if flipped[i][j] > 0 or slice[i][j] > 0:
                destination[i][j] = 255
    # now repeat the process
    # need to recalculate the COM, MBR, and area
    MBRx1, MBRy1, MBRx2, MBRy2 = 0, 0, 0, 0
    area = 0
    COM2x, COM2y = 0, 0
    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            if destination[i][j] > 0:
                area += 1
                if i < MBRy1:
                    MBRy1 = i
                if i > MBRy2:
                    MBRy2 = i
                if j < MBRx1:
                    MBRx1 = j
                if j > MBRx2:
                    MBRx2 = j
                COM2x += j
                COM2y += i
    COM2x /= area
    COM2y /= area
    # get the occupied ratio of the single fold for later, use the MBR
    old_occupied_ratio = area / ((MBRx2 - MBRx1 + 1) * (MBRy2 - MBRy1 + 1))
    # re slice the image
    destination = destination[MBRy1:MBRy2, MBRx1:MBRx2]
    # now need to handle new diagonal
    foldDirRising = None
    if (COM2x < destination.shape[1] / 2 and COM2y < destination.shape[0] / 2) or (COM2x > destination.shape[1] / 2 and COM2y > destination.shape[0] / 2):
        foldDirRising = True
    elif (COM2x > destination.shape[1] / 2 and COM2y < destination.shape[0] / 2) or (COM2x < destination.shape[1] / 2 and COM2y > destination.shape[0] / 2):
        foldDirRising = False
    else:
        print("odd...")
    flipped = None
    if foldDirRising:
        flipped = np.transpose(destination)
    else:
        flipped = np.flipud(np.transpose(destination))
    
    destination2 = np.zeros(destination.shape, dtype=np.uint8)
    cap = min(destination.shape[0], destination.shape[1])
    for i in range(cap):
        for j in range(i if foldDirRising else cap - i):
            if flipped[i][j] > 0 or destination[i][j] > 0:
                destination2[i][j] = 255
    # show the double folded image
    # img = destination2.copy()
    # cv2.imshow("double flipped", img)
    # cv2.waitKey(0)
    
    # now get MBR of destination2 and calculate occupied ratio
    f_MBRx1, f_MBRy1, f_MBRx2, f_MBRy2 = 0, 0, 0, 0
    f_area = 0
    for i in range(destination2.shape[0]):
        for j in range(destination2.shape[1]):
            if destination2[i][j] > 0:
                f_area += 1
                if i < f_MBRy1:
                    f_MBRy1 = i
                if i > f_MBRy2:
                    f_MBRy2 = i
                if j < f_MBRx1:
                    f_MBRx1 = j
                if j > f_MBRx2:
                    f_MBRx2 = j
    occupied_ratio = f_area / ((f_MBRx2 - f_MBRx1 + 1) * (f_MBRy2 - f_MBRy1 + 1))

    # print("Old occupied ratio is", old_occupied_ratio)
    # print("New occupied ratio is", occupied_ratio)
    if occupied_ratio > old_occupied_ratio and occupied_ratio > 0.6:
        # print("Building is C shaped")
        return True
    else:
        # print("Building is not C shaped")
        return False

def isIShaped(building):
    # an I shaped building will have empty space along the horizontal axis on the far left and far right
    # query these areas to see if they are empty
    # if they are, then the building is I-shaped
    comx, comy = building.COM
    leftSide = building.MBR[0]
    rightSide = building.MBR[2]
    topSide = building.MBR[1]
    bottomSide = building.MBR[3]
    comx = round(comx)
    comy = round(comy)
    img = building.img
    topOccupied = img[topSide + 1, comx] == 0
    bottomOccupied = img[bottomSide - 1, comx] == 0
    leftOccupied = img[comy, leftSide + 1] == 0
    rightOccupied = img[comy, rightSide - 1] == 0
    # print("Value of top pixel is", original_image[topSide + 1, comx])
    # print("Value of bottom pixel is", original_image[bottomSide - 1, comx])
    # print("Value of left pixel is", original_image[comy, leftSide + 1])
    # print("Value of right pixel is", original_image[comy, rightSide - 1])
    # print(f"T({comx, topSide + 1}):{topOccupied}, B({comx, bottomSide - 1}):{bottomOccupied}, L({leftSide + 1, comy}):{leftOccupied}, R({rightSide - 1, comy}):{rightOccupied}")
    ishaped = False
    # left and right check
    if leftOccupied and rightOccupied:
        # print(f"Building is I-shaped, ({leftSide}, {comy}) and ({rightSide}, {comy}) are empty (left/right check)))")
        ishaped = True
    # top and bottom check
    if topOccupied and bottomOccupied:
        # print(f"Building is I-shaped, ({comx}, {topSide}) and ({comx}, {bottomSide}) are empty (top/bottom check))")
        ishaped = True
    testimg = cv2.cvtColor(building.img.copy(), cv2.COLOR_GRAY2RGB)
    
    cv2.circle(testimg, (leftSide, comy), 0, (0, 255, 0) if leftOccupied else (255, 0, 0), 5)
    cv2.circle(testimg, (rightSide, comy), 0, (0, 255, 0) if rightOccupied else (255, 0, 0), 5)
    cv2.circle(testimg, (comx, topSide), 0, (0, 255, 0) if topOccupied else (255, 0, 0), 5)
    cv2.circle(testimg, (comx, bottomSide), 0, (0, 255, 0) if bottomOccupied else (255, 0, 0), 5)
    # cv2.imshow("testimg", testimg)
    # cv2.waitKey(0)
    
    if ishaped and building.occupied_ratio < 0.8:
        #print("Override: Building is not I-shaped (occupied ratio too low)")
        return False
    # if not ishaped:
        # print("Building is not I-shaped")
    return ishaped

def decide_size(building, calibration_data):
    # the size data is stored under calibration_data["size"]
    buildingsize = building.area
    # the sizes are smallest, small, medium, large, largest
    if buildingsize < calibration_data["size"]["smallest"][1]:
        return "smallest"
    elif buildingsize < calibration_data["size"]["small"][1]:
        return "small"
    elif buildingsize < calibration_data["size"]["medium"][1]:
        return "medium"
    elif buildingsize < calibration_data["size"]["large"][1]:
        return "large"
    else:
        return "largest"
    
def decide_aspect_ratio(building, calibration_data):
    # the aspect ratio data is stored under calibration_data["aspect"]
    buildingaspect = building.aspect
    # the aspect ratios are narrow, medium-wide, wide
    if buildingaspect < calibration_data["aspect"]["narrow"][1]:
        return "narrow"
    elif buildingaspect < calibration_data["aspect"]["medium-wide"][1]:
        return "medium-wide"
    else:
        return "wide"
    

def calibrate_where(buildings):
    # make dict into list
    buildings = list(buildings.values())
    where_calibration_data = {}
    
    # vertical ================================================================
    
    # the categories are 
    # uppermost, upper, mid-height, lower, lowermost
    # these values have been chosen to gruop the campus evenly based on low library
    mid_height_y = 221.5
    where_calibration_data["vertical"] = {}
    where_calibration_data["vertical"]["uppermost"] = [0, 60]
    where_calibration_data["vertical"]["upper"] = [61, round(mid_height_y * 0.9) - 1]
    where_calibration_data["vertical"]["mid-height"] = [round(mid_height_y * 0.9), round(mid_height_y * 1.1)]
    where_calibration_data["vertical"]["lower"] = [round(mid_height_y * 1.1) + 1, 390]
    where_calibration_data["vertical"]["lowermost"] = [0, 500]
    
    
    # horizontal ==============================================================
    # these groupings consider left and rightmost as touching or almost touching the edge of the image
    # then, mid-width is the middle 20% of the image to catch the central buildings
    # left and right capture the rest of the image
    
    
    mid_width_x = 135
    where_calibration_data["horizontal"] = {}
    where_calibration_data["horizontal"]["leftmost"] = [0, 40]
    where_calibration_data["horizontal"]["left"] = [41, round(mid_width_x * 0.9) - 1]
    where_calibration_data["horizontal"]["mid-width"] = [round(mid_width_x * 0.9), round(mid_width_x * 1.1)]
    where_calibration_data["horizontal"]["right"] = [round(mid_width_x * 1.1) + 1, 235]
    where_calibration_data["horizontal"]["rightmost"] = [0, 500]
    
    return where_calibration_data
    
def decide_vertical(building, calibration_data):
    buildingY = building.COM[1]
    if buildingY < calibration_data["vertical"]["uppermost"][1]:
        return "uppermost"
    elif buildingY < calibration_data["vertical"]["upper"][1]:
        return "upper"
    elif buildingY < calibration_data["vertical"]["mid-height"][1]:
        return "mid-height"
    elif buildingY < calibration_data["vertical"]["lower"][1]:
        return "lower"
    else:   
        return "lowermost"
    
def decide_horizontal(building, calibration_data):
    buildingX = building.COM[0]
    if buildingX < calibration_data["horizontal"]["leftmost"][1]:
        return "leftmost"
    elif buildingX < calibration_data["horizontal"]["left"][1]:
        return "left"
    elif buildingX < calibration_data["horizontal"]["mid-width"][1]:
        return "mid-width"
    elif buildingX < calibration_data["horizontal"]["right"][1]:
        return "right"
    else:
        return "rightmost"
    
    
def decide_orientation(building):
    # divide height by width, so above 1 is tall, below 1 is wide
    # tall is vertically oriented, wide is horizontally oriented
    ratio = building.MBR_height / building.MBR_width
    #print(f"Building {id_2_name(building.id)} has ratio {ratio}")
    if ratio > 1.35:
        return "vertically-oriented"
    if ratio < 0.65:
        return "horizontally-oriented"
    else:
        return "non-oriented"
    
    
# returns whether building 1 is considered near to building 2
# the larger building 1 is, the easier it is to be considered near it
def nearTo(building1, building2, tuning=0.5):
    distance = math.sqrt((building1.COM[0] - building2.COM[0])**2 + (building1.COM[1] - building2.COM[1])**2)
    a1 = building1.area
    a2 = building2.area
    # use the area of building 1 relative to the area of building 2 to encourage or discourage a nearto relationship
    if a1 > tuning*a2:
        return distance < tuning*a2
    else:
        return distance < tuning*a1