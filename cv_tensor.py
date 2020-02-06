import cv2
import numpy as np
import math
import tensorflow as tf
from utils import detector_utils as detector_utils
import keyboard

points = []
filtered = []
frame_count = 0

cap = cv2.VideoCapture('http://192.168.1.5:4747/mjpegfeed')

detection_graph, sess = detector_utils.load_inference_graph()

hand_count = 1

while True:

    ret, stream = cap.read()
    # original_stream = cv2.resize(original_stream, (320, 240))
    stream = cv2.resize(stream, (320, 240))
    stream = cv2.cvtColor(stream, cv2.COLOR_BGR2RGB)

    boxes, scores = detector_utils.detect_objects(stream, detection_graph, sess)
    lefttop, rightbottom = detector_utils.draw_box_on_image(1, 0.27, scores, boxes, 320, 240, stream)
    
    if lefttop is not None and rightbottom is not None:

        ROI = stream[lefttop[1]: lefttop[1] +  (rightbottom[1] - lefttop[1]), lefttop[0]: lefttop[0] + (rightbottom[0] - lefttop[0])]
        try:
            gray = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)
        except:
            pass
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        
        # Find Contours
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) != 0:
            cMax = max(contours, key=cv2.contourArea)
            v2.drawContours(ROI, cMax, -1, (0, 255, 0), 2)

            # Find Hull
            hull = cv2.convexHull(cMax, returnPoints=False)
            defects = cv2.convexityDefects(cMax, hull)
        
            # Find Defects
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                end = tuple(cMax[e][0])
                points.append(end)
        
            # Find Center of mass
            M = cv2.moments(cMax)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center = (cX, cY)
            cv2.circle(ROI, center, 2, (255, 0, 0), 2)

        # Filter Ends
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                try:
                    dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                except TypeError:
                    break
                if dist < 10:
                    points[j] = None
                if points[i] is not None and points[i][1] < cY:
                    filtered.append(points[i])

            if len(filtered) != 0:
                for point in filtered:
                    cv2.circle(ROI, point, 2, (0, 0, 255), 2)
                    # cv2.line(ROI, center, point, (0, 255, 0), 1)
                
        state = 0

    else:
        state = 1
        pass

    trigger = state == 1

    if trigger and frame_count % 10 == 0 and frame_count > 5:
        if state == 1:
            # keyboard.press_and_release('cmd')
            # state = 0
            pass


    cv2.imshow('Stream', stream)  

    def handle_garbage():
        if frame_count > 1:
            for i in range(5):
                points.clear()
                filtered.clear()

    handle_garbage()
     
    
    frame_count += 1  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()