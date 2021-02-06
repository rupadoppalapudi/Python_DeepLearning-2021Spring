# importing numpy
import numpy as np

# generating random vector of size 20
v = np.random.uniform(low=1, high=20, size=20)
print("\nRandom vector generated: \n", v)

# reshape the array to size 4 by 5
v1 = v.reshape((4, 5))
print("\nArray reshaped to size 4 by 5: \n", v1)

# replace the max value in each row by 0(axis1)
max_row_index = np.arange(v1.shape[0]), np.argmax(v1, axis=1)
print("\n", max_row_index)
v1[max_row_index] = 0
print("\n Array obtained after replacement: \n", v1)