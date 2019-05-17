import vlc as player
#import spotify as player
import handy
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
# capture the hand histogram by placing your hand in the box shown and
# press 'A' to confirm
# source is set to inbuilt webcam by default. Pass source=1 to use an
# external camera.
hist = handy.capture_histogram(source=0)
gestures = [0]*10
now_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # to block a faces in the video stream, set block=True.
    # if you just want to detect the faces, set block=False
    # if you do not want to do anything with faces, remove this line
    handy.detect_face(frame, block=True)
    # detect the hand
    hand = handy.detect_hand(frame, hist)
    # to get the outline of the hand
    # min area of the hand to be detected = 10000 by default
    custom_outline = hand.draw_outline(
        min_area=10000, color=(0, 255, 255), thickness=2)
    # to get a quick outline of the hand
    quick_outline = hand.outline
    # draw fingertips on the outline of the hand, with radius 5 and color red,
    # filled in.
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)
    # to get the centre of mass of the hand
    com = hand.get_center_of_mass()
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)

    y = [p[0] for p in hand.fingertips]
    x = [p[1] for p in hand.fingertips]
    if not y:
        min_y = [0]
        y = [0]
    if not x:
        min_x = [0]
        x = [0]
    min_y = max((0,min(y)-50))
    max_y = max(y)+50
    min_x = max((0,min(x)-50))
    max_x = max(x)+50
    print(min_y,max_y,min_x,max_y)
    crop_image = frame[min_x:max_x, min_y:max_y]

    cv2.imshow("Handy", quick_outline)
    cv2.imshow("Cut", crop_image)

    # display the unprocessed, segmented hand
    # cv2.imshow("Handy", hand.masked)

    # display the binary version of the hand
    # cv2.imshow("Handy", hand.binary)

     # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3,3), 0)
    
    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2,0,0]), np.array([20,255,255]))
    
    # Kernel for morphological transformation    
    kernel = np.ones((5,5))
    
    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)    
       
    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3,3), 0)
    ret,thresh = cv2.threshold(filtered, 127, 255, 0)
    
    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    
    try:
        # Find contour with maximum area
        contour = max(contours, key = lambda x: cv2.contourArea(x))
        
        # Create bounding rectangle around the contour
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Find convex hull
        hull = cv2.convexHull(contour)
        
        # Draw contour
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
        # tips) for all defects
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)

        # Print number of fingers
        # if count_defects == 0:
        #     cv2.putText(frame,"HELLO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        # elif count_defects == 1:
        #     cv2.putText(frame,"TWO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        # elif count_defects == 2:
        #     cv2.putText(frame,"THREE", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        # elif count_defects == 3:
        #     cv2.putText(frame,"FOUR", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        # elif count_defects == 4:
        #     cv2.putText(frame,"FIVE", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        # else:
        #     pass
    except:
        pass
    gestures = gestures[1:] + [count_defects]
    gesture_now = max(set(gestures), key=gestures.count)
    print(gestures)
    print(gesture_now)

    if gesture_now == 4 and not now_playing:
        player.music_toggle_play()
        now_playing = True

    if gesture_now == 1 and now_playing:
        player.music_toggle_play()
        now_playing = False

    # Show required images
    cv2.imshow("Gesture", frame)
    #all_image = np.hstack((drawing, crop_image))
    #cv2.imshow('Contours', all_image)
      
    k = cv2.waitKey(5)

    # Press 'q' to exit
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()