import cv2
import math

def mask(region):

    if len(region) != 0:
        # Performing grayscale conversion, gaussian-blur, OTSU-Thresholding
        print(region)
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            return mask
        except:
            return None

def find_max_contour(region):

    contours = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    
    if len(contours) != 0:

        #Find largest contour    
        cMax = max(contours, key=cv2.contourArea)

        return cMax
    
    else:

        return None

def plot_center_of_mass(region, plot_area, contour):

    # Find region center
    M = cv2.moments(contour)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    center = (cX, cY)
    cv2.circle(plot_area, center, 2, (0, 255, 0), 1)
    return cY

def plot_ends(region, plot_area, contour, cY):

    end_points = []
    filtered_ends = []

    # Find contour defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    
    # Find endpoints
    for i in range(defects.shape[0]):
        
        s, e, f, d = defects[i, 0]
        end = tuple(contour[e][0])
        end_points.append(end)

    # Filter and plot endpoints
    for i in range(len(end_points)):

        for j in range(i + 1, len(end_points)):

            try:
                dist = math.sqrt((end_points[i][0] - end_points[j][0]) ** 2 + (end_points[i][1] - end_points[j][1]) ** 2)
            except TypeError:
                break

            if dist < 10:
                end_points[j] = None

            if end_points[i] is not None and end_points[i][1] < cY:
                    filtered_ends.append(end_points[i])

        if len(filtered_ends) != 0:

            for point in filtered_ends:
                
                cv2.circle(plot_area, point, 2, (0, 255, 255), 2)