import numpy as np

crowd_matrix = np.loadtxt('Crowd.txt')

# get the highest 3 values in each row and sum them    
max_score = sum([sum(sorted(row)[-3:]) for row in crowd_matrix])
print(max_score)