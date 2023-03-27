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
        if building.MBR[1] >= b.MBR[3] or building.MBR[3] <= b.MBR[1] or building.MBR[0] >= b.MBR[2] or building.MBR[2] <= b.MBR[0]:
            continue
        else:
            intersections.append(b)
    return intersections