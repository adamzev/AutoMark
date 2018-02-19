import cv2
import numpy as np
import pickle
import copy

from helpers import Helpers
 
helpers = Helpers()
img = helpers.loadImage('start.jpg')

z_t = 1
z_s = 0
P = np.zeros((3,3))

P[2, 2] = 1

t = np.zeros((4,3))
s = np.zeros((4,3))

# corners
t[0] = np.array([0,  55, 0]) # top left
t[1] = np.array([209,  0, 0]) # top right
t[2] = np.array([354, 230, 0]) # bottom left
t[3] = np.array([54,  354, 0]) # bottom right

t0 = copy.copy(t[0])
for i in range(0, len(t)):
    t[i] = t[i] - t0
    t[i][2] = z_t # make all z's into 1

# target
s[0] = np.array([0,  0, z_s]) # top left
s[1] = np.array([220,  0, z_s]) # top right
s[2] = np.array([220, 220, z_s]) # bottom left
s[3] = np.array([0,  220, z_s]) # bottom right

xt = 0 # is the mean of the differences between the image and the point

A = []
for i in range(len(t)):
    x, y, z = t[i]
    a, b, c = s[i]

    A.append([x, y, z, 0, 0, 0, -1*a*x, -1*a*y, -1*a])
    A.append([0, 0, 0, x, y, z, -1*b*x, -1*b*y, -1*b])

zero = np.zeros(len(t)*2)
A = np.asarray(A)
U, S, Vh = np.linalg.svd(A)
L = Vh[-1,:] / Vh[-1,-1]
h = L.reshape(3, 3)
#h = np.linalg.lstsq(A, zero, rcond=None)
#h = list(h[3])
#h.append(1)
#h = np.array(h)
#h = h.reshape(3,3)
print(h)
#h, status = cv2.findHomography(t, s)
print()
print(h)
final_img = cv2.warpPerspective(img, h, (400, 400))

helpers.show(img)
helpers.show(final_img)
'''
difs = s - t
print(difs)
print(difs.mean(axis=0))
xt = difs[0]
yt = difs[1]
for i in range(len(t)):
    print(i)
'''