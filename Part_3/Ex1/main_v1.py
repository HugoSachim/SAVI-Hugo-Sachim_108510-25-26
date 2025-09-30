#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
import cv2  # import the opencv library
import numpy as np
import argparse


def main():

    # ------------------------------------
    # Setu pargparse
    # ------------------------------------
    parser = argparse.ArgumentParser(
        prog='Traffic car couter',
        description='Counts cars',
        epilog='This is finished')

    parser.add_argument('-if', '--input_filename', type=str, default='/home/hogu/Desktop/SAVI-Hugo-Sachim_108510-25-26/Part_3/Ex1/traffic.mp4')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # Open the video file
    # -------------------------------------
    capture = cv2.VideoCapture(args['input_filename'])

    # Check if the sequence was opened successfully
    if not capture.isOpened():
        print("Error: Could not open image sequence.")
    else:
        print("Image sequence opened successfully!")

    # ------------------------------------
    # Read and display all frames
    # ------------------------------------

    width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(height,width)
    line_1 = width/4
    line_2 = (width/4)*2
    line_3 = (width/4)*3
    line_4 = (width/4)*4
    
    last_1_avg= 60
    last_2_avg = 60
    last_3_avg = 60
    last_4_avg = 60

    cars_lane_1 = 0
    cars_lane_2 = 0
    cars_lane_3 = 0
    cars_lane_4 = 0

    thre = 25


    while True:

        ret, frame = capture.read()
        if ret == False:
            break

        cv2.imshow('Current frame', frame)

        # Break if q is pressed
        key = cv2.waitKey(20)
        # print('key = ' + str(key))
        if key == 113:
            print('You pressed q. Quitting.')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_line = gray[-1, :] 
        #width_line= last_line.shape[0]
        #print(width_line)

        line_1_avg_segment = last_line[0:int(float(line_1))]
        line_1_avg = np.mean(line_1_avg_segment)

        # Take the last row, but only pixels 200 to 500
        #segment = gray[-1, 200:501]   # 200 to 500 inclusive

        # Compute the average
        #avg_value = np.mean(segment)


        line_2_avg_segment = last_line[int(float(line_1)):int(float(line_2))]
        line_2_avg = np.mean(line_2_avg_segment)


        line_3_avg_segment = last_line[int(float(line_2)):int(float(line_3))]
        line_3_avg = np.mean(line_3_avg_segment)


        line_4_avg_segment = last_line[int(float(line_3)):int(float(line_4))]
        line_4_avg = np.mean(line_4_avg_segment)

        #print(line_3_avg)

        diif_1 = abs(line_1_avg-last_1_avg)
        diif_2 = abs(line_2_avg-last_2_avg)
        diif_3 = abs(line_3_avg-last_3_avg)
        diif_4 = abs(line_4_avg-last_4_avg)
        
        #print(diif_1)
        
        if diif_1 > thre:
            cars_lane_1 += 1

        if diif_2 > thre:
            cars_lane_2 += 1
        
        if diif_3 > thre:
            cars_lane_3 += 1

        if diif_4 > thre:
            cars_lane_4 += 1
        
        last_1_avg = line_1_avg
        last_2_avg = line_2_avg
        last_3_avg = line_3_avg
        last_4_avg = line_4_avg

        #print(diif_3)
        print(cars_lane_3)

    cv2.destroyAllWindows()  # Close the window


if __name__ == '__main__':
    main()
