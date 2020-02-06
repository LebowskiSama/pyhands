import cv2
import numpy as np
import math

HSVlower = np.array([0, 0, 0])
HSVupper = np.array([180, 255, 128])

points = []
filtered = []
frame_count = 0

cap = cv2.VideoCapture('http://192.168.1.2:4747/mjpegfeed')


while True:

    ret, stream = cap.read()
    stream = cv2.resize(stream, (320, 240))

    hsv = cv2.cvtColor(stream, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv, HSVlower, HSVupper)

    # Find Contours
    contours = cv2.findContours(hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    if len(contours) != 0:
        
        cMax = max(contours, key=cv2.contourArea)
        # cv2.drawContours(stream, cMax, -1, (0, 255, 0), 2)
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
        cv2.circle(stream, center, 2, (255, 0, 0), 2)

        # Filter Ends
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                try:
                    dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                except TypeError:
                    break
                if dist < 50:
                    points[j] = None
                if points[i] is not None and points[i][1] < cY:
                    filtered.append(points[i])

        print(filtered)

        if len(filtered) != 0:
            for point in filtered:
                cv2.circle(stream, point, 2, (0, 0, 255), 2)
                cv2.line(stream, center, point, (0, 255, 0), 2)
                

    def handle_garbage():
        if frame_count > 1:
            for i in range(5):
                points.clear()
                filtered.clear()

    handle_garbage()
    cv2.imshow('Stream', stream)
    frame_count += 1  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()