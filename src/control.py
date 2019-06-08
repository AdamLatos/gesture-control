#import Vlc as player
import Spotify as player
#import control_helpers
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

DEBUG = 0
gestures = [0]*10
gesture_last = 0
now_playing = False
skin_hue = 10
cx = 0
cy = 0
cx_last = 0
cy_last = 0
volup_cnt = 0
voldown_cnt = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
back_sub = cv2.createBackgroundSubtractorKNN(history = 1000, detectShadows=False)

def value_check(event, x, y, flags, param):
    global hsv, boundsq, hsv_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"At {x},{y} value is {hsv[(y,x)]}")
        print(f"Bounds are {bounds}")
        print(f"HSV mask is {hsv_mask[(y,x)]}")

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)

def print_hist(hist_img):
    hist=np.int32(np.around(hist_img))
    h = np.zeros((255,hist_bins,3), np.uint8)
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y

def filter_points(points, filter_value):
    # unused for now
    for i in range(len(points)):
        for j in range(i + 1, len(points)):    
            if points[i] and points[j] and dist(points[i], points[j]) < filter_value:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered

face_detected = 0

# BEGIN CAPTURE

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break
    orig = img.copy()

    # Detect face using haar cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if faces == () and not face_detected:
        continue
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)
        cv2.rectangle(img,(x,y),(x+w,y+h*2),(0,0,0),-1)

    face_detected = 1
    #cv2.imshow('img',img)

    # alternative binarization
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    ###

    hist_bins = 180
    hist_img = cv2.calcHist([roi_hsv], [0], None, [hist_bins], [0, 180])
    cv2.normalize(hist_img, hist_img, 0, 255, cv2.NORM_MINMAX)

    if DEBUG:
        cv2.imshow("histogram", print_hist(hist_img))

    skin_hue = np.argmax(hist_img[0:40]) if np.argmax(hist_img[0:40]) else skin_hue

    thr = 15
    hue_lb = skin_hue - thr
    hue_hb = skin_hue + thr
    s_lb = 0
    v_lb = 120
    bounds = (0,180)
    if hue_lb < 0:
        lb1 = np.array([0, s_lb, v_lb], np.uint8)
        ub1 = np.array([hue_hb, 255, 255], np.uint8)    
        lb2 = np.array([180 - hue_hb, s_lb, v_lb], np.uint8)
        ub2 = np.array([180, 255, 255], np.uint8)
        mask1 = cv2.inRange(hsv, lb1, ub1)
        mask2 = cv2.inRange(hsv, lb2, ub2)
        hsv_mask = cv2.bitwise_or(mask1,mask2)
        bounds = (lb1, ub1, lb2, ub2)
    else:
        lb = np.array([hue_lb, s_lb, v_lb], np.uint8)
        ub = np.array([hue_hb, 255, 255], np.uint8)
        hsv_mask = cv2.inRange(hsv, lb, ub)
        bounds = (lb, ub)

    kernel = np.ones((3,3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    # hsv_mask = cv2.dilate(hsv_mask, kernel, iterations = 1)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    # hsv_mask = cv2.erode(hsv_mask, kernel, iterations = 1)

    if DEBUG:
        cv2.imshow("hsv_mask", hsv_mask)

    # Extract foreground
    kernel = np.ones((3,3))
    front_mask = back_sub.apply(img)
    front_mask = cv2.dilate(front_mask, kernel, iterations = 1)
    front_mask = cv2.erode(front_mask, kernel, iterations = 1)  
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)

    if DEBUG:
        cv2.imshow("front_mask", front_mask)

    mask = cv2.bitwise_and(hsv_mask, front_mask)
    
    if DEBUG:
        cv2.imshow("mask", mask)

    kernel = np.ones((3,3))
    dilation = cv2.dilate(mask, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)

    # FIND CONTOURS:

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 3000
    palm_area = 0
    flag = None
    cnt = None
    defects_cnt = 0
    points_end = []
    points_far = []
    points_start = []
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > palm_area:
            palm_area = area
            flag = i
    if flag is not None and palm_area > min_area:
        cnt = contours[flag]
        cv2.drawContours(img, [cnt], 0, (255,0,0), 2)
        hull_pts = cv2.convexHull(cnt)
        hull = cv2.convexHull(cnt, returnPoints = False)
        if(hull.size > 3):
            defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            #cv2.line(img,start,end,[0,255,0],2)
            #cv2.circle(img,far,5,[0,0,255],-1)
            points_end.append(end)
            points_far.append(far)
            points_start.append(start)

        # not filtering for now, better results
        filtered_start = points_start
        filtered_end = points_end
        filtered_far = points_far

        # filtered_star = filter_points(points_start, 30)
        # filtered_end = filter_points(points_end,30)
        # filtered_far = filter_points(points_far,30)
        # fingertips = [pt for idx, pt in zip(range(5), filtered)]
        # for fi in filtered_end:
        #     cv2.circle(img,fi,5,[0,0,255],-1)
        # for fi in filtered_far:
        #     cv2.circle(img,fi,5,[0,100,255],-1)
        # for fi in filtered_start:
        #     cv2.circle(img,fi,5,[100,0,255],-1)

        # centroid:
        M = cv2.moments(hull_pts)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img,(cx,cy),10,[255,0,0],-1)

        # checking number of fingers:
        for i in range(len(filtered_end)):
            start = filtered_start[i]
            end = filtered_end[i]
            far = filtered_far[i]

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # if angle <= 90 draw a circle at the far point
            if angle <= 90:
                defects_cnt += 1
                cv2.circle(img,far,5,[0,0,255],-1)
            cv2.line(img,start,end,[0,255,0],2)
            # Print number of fingers

    gestures = gestures[1:] + [defects_cnt]
    gesture_now = max(set(gestures), key=gestures.count)
    print(gestures)
    print(gesture_now)

    if gesture_now >= 4 and not now_playing:
        player.music_toggle_play()
        now_playing = True

    if gesture_now == 1 and now_playing:
        player.music_toggle_play()
        now_playing = False

    if (gesture_now == 2 and gesture_last == 2) or (gesture_now == 3 and gesture_last == 3) and now_playing:
        if cy_last < cy:
            voldown_cnt += 1
            volup_cnt = 0
        elif cy_last > cy:
            voldown_cnt = 0
            volup_cnt += 1
        if volup_cnt > 3:
            player.music_vol_up()
            volup_cnt = 0
        elif voldown_cnt > 3:
            player.music_vol_down()
            voldown_cnt = 0

    print(volup_cnt, voldown_cnt)

    gesture_last = gesture_now
    cy_last = cy
    cx_last = cx


    cv2.imshow('img', img)
    cv2.imshow('end', thresh)
    cv2.setMouseCallback("hsv_mask", value_check)

    k = cv2.waitKey(5)
    if k == ord('q'):
        break
    if cv2.getWindowProperty('end', 0) < 0:
        break
cap.release()
cv2.destroyAllWindows()