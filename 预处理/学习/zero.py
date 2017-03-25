import numpy as np

labels = []
l = np.array([1,2,2,1])

for index in l:
    a = np.zeros(10, dtype=np.int32)
    a[index] = 1



    labels.append(a)
print (labels)