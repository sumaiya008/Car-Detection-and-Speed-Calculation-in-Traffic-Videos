# importing libraries
import cv2
import numpy as np

# capturing or reading video
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./Car_data/video.avi')

count_line_position = 1000

# minimum contour width
min_contour_width=40  #40
# minimum contour height
min_contour_height=40  #40


# Initial subtractor
algo = cv2.createBackgroundSubtractorMOG2()

# defining a function
def get_centroid(x, y, w, h):

    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx,cy
    return [cx, cy]

detect = []
offset=6 #allowable error between pixel
counter = 0
while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contourSahpe, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (5, count_line_position), (1900, count_line_position), (255,127,0), 1)
    
    for(i,c) in enumerate(contourSahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame1, "Vehicles" + str(counter), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 244, 0), 2)
        
        centroid = get_centroid(x, y, w, h)
        detect.append(centroid)
        cv2.circle(frame1,centroid, 4, (0,255,0), -1)
        
        #cx,cy= get_centroid(x, y, w, h)
        for (x,y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter+=1
                cv2.line(frame1, (5, count_line_position), (1900, count_line_position), (0,127,255), 1)
                detect.remove((x,y))
                print("Vehicles counter:" +str(counter))
            
    cv2.putText(frame1, "Total Vehicles Detected: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)
    
    #cv2.imshow("OUTPUT", dilatada)
    cv2.imshow('original',frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
