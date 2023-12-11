# importing libraries
import cv2
import numpy as np
import time


# capturing or reading video
# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./Car_data/video.avi')
cap = cv2.VideoCapture('cars.mp4')

count_line_position = 600
# minimum contour width
min_contour_width=30 
# minimum contour height
min_contour_height=30 

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

# Adding the following variables for speed calculation
fps_start_time = time.time()
fps = 0
total_frames = 0
speed_kph = 0
start_point = (0,0)

def calculate_speed(start_point, end_point, fps):
    distance_pixels = abs(end_point[0] - start_point[0])
    distance_meters = distance_pixels / pixels_per_meter  # Adjust 
    speed_mps = distance_meters / fps
    speed_kph = speed_mps * 3.6
    return speed_kph

pixels_per_meter = 10

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
        #cv2.putText(frame1, "Vehicles" + str(counter), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 244, 0), 2)
        
        centroid = get_centroid(x, y, w, h)
        detect.append(centroid)
        cv2.circle(frame1,centroid, 4, (0,255,0), -1)
        
        #cx,cy= get_centroid(x, y, w, h)
        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                cv2.line(frame1, (5, count_line_position), (700, count_line_position), (0, 127, 255), 1)
                detect.remove((x, y))
                print("Vehicle Count: " + str(counter))

                # Calculate speed
                end_point = (x, y)
                speed_kph = calculate_speed(start_point, end_point, fps)
                print(f"Speed of vehicle {counter}: {speed_kph:.2f} km/h")

        cv2.putText(frame1, f"Speed of Vehicle{counter}:{speed_kph:.2f} km/h", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

                
    # Update fps and total frames
    total_frames += 1
    fps_end_time = time.time()
    fps = total_frames / (fps_end_time - fps_start_time)

    # Display speed and fps
    #cv2.putText(frame1, f"Speed: {speed_kph:.2f} km/h", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #cv2.putText(frame1, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('original', frame1)
    if cv2.waitKey(1) == 13:
        break
    
cv2.destroyAllWindows()
cap.release()
