# Imports
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from collections import defaultdict

# is_int(value) takes in a value and attempts to type cast it
# to an int.  If it fails, it returns false.  Otherwise, true.
def is_int(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

# segment_by_angle_kmeans(lines, k, kwargs) seperates the filtered lines (lines) 
# by orientation - horizontal and vertical.
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # Multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # Run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # Segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

# find_intsersection(line_1, line_2) finds the intersection point between the
# two given lines.
def find_intersect(line_1, line_2):
    rho_1, theta_1 = line_1
    rho_2, theta_2 = line_2
    a = np.array([[np.cos(theta_1), np.sin(theta_1)], [np.cos(theta_2), np.sin(theta_2)]])
    b = np.array([[rho_1], [rho_2]])
    if a[0][0] == a[1][0] or a[0][1] == a[1][1]:
        return[[-1, -1]]
    x, y = np.linalg.solve(a, b)
    x, y = int(np.round(x)), int(np.round(y))
    return [[x, y]]

# get_array() returns a 2D list with the values of the read Sudoku puzzle.
def get_array(imgpath):
    # Step 1: Load image, add a border, greyscale.
    image = cv2.imread(imgpath)
    WHITE = [255, 255, 255]
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
    rgbimage = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((image.shape), np.uint8)
    kernel = np.ones((5,5),np.uint8)

    #cv2.imwrite('image.png', image) #Testing

    # Step 2: Hough transform
    edges = cv2.Canny(image,90,150,apertureSize = 3)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 1)

    sensitivity = 135

    # We attempt try until we get 20 to 21 lines.  This is likely to be changed
    # in the future, as it's buggy and not reliable (but works... sometimes).
    while True:
        lines = cv2.HoughLines(edges,1,np.pi/180,sensitivity)

        rho_threshold = 15
        theta_threshold = 0.1

        # Look for lines that are similar.
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # We now order the indices of the lines by how may other lines are similar.
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # Base for filtering.
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:
                continue

            for j in range(i + 1, len(lines)):
                if not line_flags[indices[j]]:
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False

        #print('number of Hough lines:', len(lines)) #For testing

        filtered_lines = []
        if filter:
            for i in range(len(lines)): # filtering
                if line_flags[i]:
                    filtered_lines.append(lines[i])
            #print('Number of filtered lines:', len(filtered_lines)) #For testing
        else:
            filtered_lines = lines

        # Adjust sensitivity.  As mentioned above, likely to change strategies.
        if len(filtered_lines) <= 21 and len(filtered_lines) >= 20:
            break
        elif len(filtered_lines) > 21:
            sensitivity += 1
        elif len(filtered_lines) < 20:   
            sensitivity -= 1

    # Step 3: We now segment the lines based on orientation.
    segmented = segment_by_angle_kmeans(filtered_lines)

    # Step 4: Draw lines.
    for i in range(0, 2):
        for line in segmented[i]:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            if i == 0:
                #cv2.line(rgbimage, (x1, y1), (x2, y2), (0, 0, 255), 2) #Draw line in RGB, for testing
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) #Draw line
            elif i == 1:
                #cv2.line(rgbimage, (x1, y1), (x2, y2), (0, 255, 0), 2) #Draw line in RGB, for testing
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) #Draw line

    # Step 5: Find intersections between lines, and sort from top left to bottom right.
    inter_pt = []
    #print ("SEG:", segmented)
    #print("FILTER: ", filtered_lines)
    for line_1 in segmented[0]:
        for line_2 in segmented[1]:
            temp = find_intersect(line_1[0], line_2[0])
            h,w = np.shape(image)
            if temp[0][1] >= 0 and temp[0][0] >= 0 and temp[0][1] <= h and temp[0][0] <= w:
                inter_pt.append(temp)
    inter_pt = sorted(inter_pt, key=lambda k: [k[0][1], k[0][0]])
    print("LEN: ", len(inter_pt)) #Testing

    # Step 6 (subject to removal): Reduce excess points.
    if len(inter_pt) > 100:
        #print("PREV LEN: ", len(inter_pt)) #Testing
        left_top = inter_pt[0][0]
        right_bot = inter_pt[-1][0]
        sqr_x = int((right_bot[0] + left_top[0]) / 9)
        sqr_y = int((right_bot[1] + left_top[1]) / 9)
        #print ("X: ", sqr_x,", Y: ", sqr_y) #Testing
        prev_cord = [-9000, -9000] #Very, VERY hacky.
        for i in inter_pt[:]:
            x1 = i[0][0]
            y1 = i[0][1]
            if x1 - prev_cord[0] < sqr_x or y1 - prev_cord[1] < sqr_y:
                inter_pt.remove(i)
            else:
                prev_cord = [x1, y1]
    #print ("INTER AFTER AFTER: ", inter_pt) # SHOULD BE 100.
    #print("Intersect:", len(inter_pt))
    #for i in inter_pt:
    #    x1 = i[0][0]
    #    y1 = i[0][1]
    #    cv2.line(rgbimage, (x1, y1), (x1, y1), (255, 0, 0), 3) #Draw intersection in RGB, for testing
    #cv2.imwrite('hough.jpg', rgbimage) #Testing

    # Step 7: Invert the colours of the Sudoku puzzle, along with the new lines.
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)
    filename = "{}sudo_temp.png".format(os.getpid())

    # Step 8: Traverse through the intersecting points to form squares, then take images of
    # each square and use libtesseract to determine the value of each square.  We save to a text file.
    sudoku = []
    for i in range(0, 9):
        temp = []
        for j in range(0, 9):
            x1 = inter_pt[i*10+j][0][0]
            y1 = inter_pt[i*10+j][0][1]
            x2 = inter_pt[(i+1)*10+(j+1)][0][0]
            y2 = inter_pt[(i+1)*10+(j+1)][0][1]
            #Testing
            #cv2.rectangle(rgbimage, (x1,y1), (x2,y2), (0,255,0), 3)
            #cv2.imwrite("rect.png", rgbimage)
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,0), 3)
            crop_img = image[y1:y2, x1:x2]
            crop_img = cv2.bitwise_not(crop_img)
            crop_img = cv2.copyMakeBorder(crop_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=WHITE)
            cv2.imwrite(filename, crop_img)
            text = pytesseract.image_to_string(Image.open(filename), lang='eng', boxes=False, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            if is_int(text):
                temp.append(int(text))
                #print(int(text))
            else:
                temp.append(0)
                #print(0)
        sudoku.append(temp)
        os.remove(filename)
    finalimg = "{}sudo_temp.png".format(os.getpid())
    finaloutput = "sudo_out.txt"
    file = open(finaloutput, "w")
    for i in sudoku:
        file.write(str(i))
        file.write('\n')
    file.close()
    return sudoku
