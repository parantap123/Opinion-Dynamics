import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

n = 100

def makeGaussian(size, fwhm = 40, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    x0 = y0 = size // 2

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)

graph = np.ones((100, 100))
dist = makeGaussian(n)
graph = np.random.rand(100, 100)
graph = (graph < dist) * 1

plt.imshow(graph)
plt.show()
print(graph)
