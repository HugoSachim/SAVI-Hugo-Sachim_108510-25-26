#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
import glob
import cv2  # import the opencv library
import numpy as np
import argparse

def bbox_car_per_lane(car_lane, bbox):
    bbox['x2'] = bbox['x'] + bbox['w']
    bbox['y2'] = bbox['y'] + bbox['h']
    bbox['car_lane'] = car_lane

    #average to calculate pixel of the matrix
    bbox['previous_average'] = None
    bbox['change_event'] = False

    return bbox

def draw_rect_per_lane(bbox,frame_gui):
    # Draw the box on the image
    cv2.rectangle(frame_gui, (bbox['x'], bbox['y']), (bbox['x2'], bbox['y2']),
                    (255, 0, 0), 2)    
    return frame_gui

def chage_event_per_lane(frame_gui, frame_gray, bbox):
    box_values = frame_gray[bbox['y']:bbox['y2'],  bbox['x']:bbox['x2']]
    # print('box_values = ' + str(box_values))

    average = round(np.mean(box_values), 1)  # type: ignore
    # print('average = ' + str(average))
    cv2.putText(
        frame_gui, 'mean ' + str(average),
        (bbox['x'],
        bbox['y'] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Compute the difference between the previous and the current
    if bbox['previous_average'] is None:
        difference = 0  # TODO João has questions
    else:
        difference = round(abs(average - bbox['previous_average']), 1)

    bbox['previous_average'] = average  # update previous average
    cv2.putText(
        frame_gui, 'dif ' + str(difference),
        (bbox['x'],
        bbox['y'] - 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Change detection event
    change_detection_threshold = 10
    if difference > change_detection_threshold:
        bbox['change_event'] = True
    else:
        bbox['change_event'] = False

    # Selecting color for putting the change event text as a function of the value
    if bbox['change_event'] == True:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    cv2.putText(
        frame_gui, 'event ' + str(bbox['change_event']),
        (bbox['x'],
        bbox['y'] - 70),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return frame_gui, frame_gray, bbox

def count_car_per_lane(bbox):
    print('Hello World')
    #     # Option 2
    # if previous_change_event_4 == False and previous_change_event_3 == False and previous_change_event_2 == False and previous_change_event_1 == False and previous_change_event == False and change_event == True:  # a rising edge
    #     number_of_cars += 1

    # previous_change_event_4 = previous_change_event_3
    # previous_change_event_3 = previous_change_event_2
    # previous_change_event_2 = previous_change_event_1
    # previous_change_event_1 = previous_change_event
    
    # previous_change_event = change_event

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

    # Define bbox coordinates
    bbox_lane_1_coord = {'x': 190, 'y': 470, 'w': 100, 'h': 200}
    bbox_lane_2_coord = {'x': 420, 'y': 470, 'w': 100, 'h': 200}
    bbox_lane_3_coord = {'x': 740, 'y': 470, 'w': 100, 'h': 200}
    bbox_lane_4_coord = {'x': 990, 'y': 470, 'w': 100, 'h': 200}
    
    bbox_lane_1 = bbox_car_per_lane(1, bbox_lane_1_coord)
    bbox_lane_2 = bbox_car_per_lane(2, bbox_lane_2_coord)
    bbox_lane_3 = bbox_car_per_lane(3, bbox_lane_3_coord)
    bbox_lane_4 = bbox_car_per_lane(4, bbox_lane_4_coord)

    # print(bbox_lane_1)
    # print(bbox_lane_2 )
    # print(bbox_lane_3)
    # print(bbox_lane_4)

    # ------------------------------------
    # Read and display all frames
    # ------------------------------------
    previous_average = None
    previous_change_event = False
    previous_change_event_1 = False
    previous_change_event_2 = False
    previous_change_event_3 = False
    previous_change_event_4 = False

    total_number_of_cars = 0
    frame_count = 0
    while True:

        ret, frame = capture.read()
        if ret == False:
            break

        # create image for drawing
        frame_gui = deepcopy(frame)

        # Draw the frame count
        cv2.putText(
            frame_gui, '#frame ' + str(frame_count),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the box on the image
        draw_rect_per_lane(bbox_lane_1,frame_gui)
        draw_rect_per_lane(bbox_lane_2,frame_gui)
        draw_rect_per_lane(bbox_lane_3,frame_gui)
        draw_rect_per_lane(bbox_lane_4,frame_gui)

        # cv2.imshow('Image GUI', frame_gui)
        # cv2.waitKey(0)
        # exit(0)

        # Compute grayscale image for analysing
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------------------
        chage_event_per_lane(frame_gui, frame_gray, bbox_lane_1)
        chage_event_per_lane(frame_gui, frame_gray, bbox_lane_2)
        chage_event_per_lane(frame_gui, frame_gray, bbox_lane_3)
        chage_event_per_lane(frame_gui, frame_gray, bbox_lane_4)

        # ---------------------------
        # Count a car
        # ---------------------------
        # Option 1: Count the rising edges
        #if previous_change_event == False and change_event == True:  # a rising edge
        #    number_of_cars += 1 

        # Option 2
        # if previous_change_event_4 == False and previous_change_event_3 == False and previous_change_event_2 == False and previous_change_event_1 == False and previous_change_event == False and change_event == True:  # a rising edge
        #     number_of_cars += 1

        # previous_change_event_4 = previous_change_event_3
        # previous_change_event_3 = previous_change_event_2
        # previous_change_event_2 = previous_change_event_1
        # previous_change_event_1 = previous_change_event
        
        # previous_change_event = change_event
        
        #////////////////////////-------------------------////////////////////////////
        # meter formula do count_car_per_lane()

        #nao esquecer
        #/////////////////////////----------------------//////////////////////

        # Draw the number of cars
        cv2.putText(
            frame_gui, '#cars ' + str(total_number_of_cars),
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw image
        cv2.imshow('Image GUI', frame_gui)
        # cv2.imshow('Image gray', frame_gray)

        # Break if q is pressed
        key = cv2.waitKey(0)
        # print('key = ' + str(key))
        if key == 113:
            print('You pressed q. Quitting.')
            break

        frame_count += 1  # update the frame count

    cv2.destroyAllWindows()  # Close the window


if __name__ == '__main__':
    main()
