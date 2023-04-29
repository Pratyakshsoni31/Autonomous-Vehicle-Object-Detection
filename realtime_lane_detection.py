import cv2
import numpy as np

net = cv2.dnn.readNet("C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.weights", 
                      "C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.cfg")

canny_threshold1 = 50
canny_threshold2 = 150
hough_threshold = 50
min_line_length = 100
max_line_gap = 5

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imshow('Lane Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
