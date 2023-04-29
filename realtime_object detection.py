import cv2
import numpy as np

alpha = 1.0 
beta = 0

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.weights", 
                      "C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.cfg")

canny_threshold1 = 50
canny_threshold2 = 150
hough_threshold = 50
min_line_length = 100
max_line_gap = 5

classes = []
with open("C:\\Users\\Pratyaksh\\Downloads\\yolo\\\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = [net.getLayerNames()[i-1] for i in net.getUnconnectedOutLayers()]

confidence_threshold = 0.5
nms_threshold = 0.4

conf_threshold = 0.5

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == "person" and confidence > confidence_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

            elif (classes[class_id] == "car" or classes[class_id] == "truck") and confidence > conf_threshold:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y-5), font, 1, color, 2)

    cv2.imshow("Grayscale Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()