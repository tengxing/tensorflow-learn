import random
from matplotlib import pyplot

random_walk = [-1 if random.random() < 0.5 else 1]

for i in range(1, 1000):
    random_walk.append(random_walk[i - 1] + (-1 if random.random() < 0.5 else 1))

pyplot.plot(random_walk)
pyplot.show()