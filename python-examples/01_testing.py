import numpy as np
import time

T1_array = [5,6,7]
T2_array = [3,4,5]
T3_array = [6,4,2]

arr = np.asarray([T1_array, T2_array, T3_array])
np.savetxt('sample.csv',arr, delimiter=",")
