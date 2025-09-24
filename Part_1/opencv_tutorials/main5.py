#!/usr/bin/env python3


import numpy as np
import cv2 as cv

img = cv.imread('photo.png')
assert img is not None, "file could not be read, check with os.path.exists()"

px = img[100,100]
print( px )


# accessing only blue pixel
blue = img[100,100,0]
print( blue )

img[100,100] = [255,255,255]
print( img[100,100] )
cv.rectangle(img,(20,20),(150,150),(0,255,0),0)

print(img.shape)
print(img.dtype)

cv.imshow("Display window", img)
k = cv.waitKey(0)
