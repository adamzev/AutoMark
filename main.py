import numpy as np

x = np.array([
    [0, 1, 1], [1, 1, 0], [1, 0, 1]
])

y = np.array([
    3.66, 1.55, 3.42
])

scalars = np.linalg.solve(x, y)
print(scalars)
