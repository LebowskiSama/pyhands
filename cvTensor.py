import cv2
import numpy as np
import tensorflow as tf
from utils import detector_utils as detector_utils
import handUtils
import keyboard
import serial

frame_count = 0
# uno = serial.Serial('com3', 9600, timeout=1)

cap = cv2.VideoCapture('http://192.168.1.4:4747/mjpegfeed')

detection_graph, sess = detector_utils.load_inference_graph()

hand_count = 1

while True:

    state = 0

    ret, original_stream = cap.read()
    # original_stream = cv2.resize(original_stream, (320, 240))
    original_stream = cv2.resize(original_stream, (320, 240))
    stream = cv2.cvtColor(original_stream, cv2.COLOR_BGR2RGB)

    boxes, scores = detector_utils.detect_objects(original_stream, detection_graph, sess)
    lefttop, rightbottom = detector_utils.draw_box_on_image(1, 0.27, scores, boxes, 320, 240, original_stream)

    if lefttop is not None and rightbottom is not None and frame_count > 5:
        
        ROI = original_stream[lefttop[1]: lefttop[1] +  (rightbottom[1] - lefttop[1]), lefttop[0]: lefttop[0] + (rightbottom[0] - lefttop[0])]
        
        if len(ROI) != 0:

            mask = handUtils.mask(ROI) # Masking
            cMax = handUtils.find_max_contour(mask) # Find largest bounded-contour
            # cv2.imshow('Mask', mask)

        if cMax is not None:
            com_height = handUtils.plot_center_of_mass(mask, ROI, cMax) # Plot center of hand
            handUtils.plot_ends(mask, ROI, cMax, com_height) # Plot filtered fingertips

    else:
        # uno.write(b'1')
        pass


    cv2.imshow('Stream', original_stream)  

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()