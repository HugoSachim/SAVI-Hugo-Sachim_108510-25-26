#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
import cv2  # import the opencv library
import numpy as np

# main function, where our code should be


def main():
    print("python main function")

    # --------------------------
    # Read image
    # --------------------------
    image_filename = '/home/hogu/Desktop/SAVI-Hugo-Sachim_108510-25-26/Part_2/images/scene.png'
    image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    H, W, numchannels = image.shape

    template_filename = '/home/hogu/Desktop/SAVI-Hugo-Sachim_108510-25-26/Part_2/images/wally.png'
    template = cv2.imread(template_filename, cv2.IMREAD_COLOR)
    h, w, numchannels = template.shape

    # Apply template Matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    print('result = ' + str(result))
    print('result type ' + str(result.dtype))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print('max_loc' + str(max_loc))

    # Draw a rectange on the original image
    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)

    #cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 3)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
    cv2.imshow('Result', mask)
    cv2.waitKey(0) 

    result_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_gray_bgr = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)

    wally = cv2.bitwise_and(image, image, mask=mask) 
    cv2.imshow('Result', wally)
    cv2.waitKey(0) 

    inverse_mask = cv2.bitwise_not(mask)
    cv2.imshow('Result', inverse_mask)
    cv2.waitKey(0) 

    background = cv2.bitwise_and(result_gray_bgr, result_gray_bgr, mask=inverse_mask)  
    cv2.imshow('Result', background)
    cv2.waitKey(0) 

    result_high = cv2.add(wally, background)  
    cv2.imshow('Result', result_high)
    cv2.waitKey(0) 


    cv2.imshow('Scene', image)
    cv2.imshow('Template', template)
    cv2.imshow('Result', result)
    cv2.imshow('Result', result_high)
    cv2.waitKey(25)  #

    cv2.waitKey(0)  # 0 means wait forever until a key is pressed


if __name__ == '__main__':
    main()
