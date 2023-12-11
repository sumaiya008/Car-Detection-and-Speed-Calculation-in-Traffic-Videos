# importing libraries
import cv2
import numpy as np

# capturing or reading video
cap = cv2.VideoCapture('cars.mp4')

# Get video properties (width, height, and frames per second)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for H.264 codec
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Initial subtractor
algo = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contourSahpe = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Save the processed frame to the output video
    out.write(dilatada)

    cv2.imshow("OUTPUT", dilatada)

    if cv2.waitKey(1) == 13:
        break

# Release the VideoWriter and VideoCapture objects
out.release()
cv2.destroyAllWindows()
cap.release()
