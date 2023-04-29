import cv2
import numpy as np

net = cv2.dnn.readNet("C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.weights", 
                      "C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.cfg")

classes = []
with open("C:\\Users\\Pratyaksh\\Downloads\\yolo\\\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

conf_threshold = 0.5
nms_threshold = 0.4

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == "car" or classes[class_id] == "truck":

                if confidence > conf_threshold:

                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")

                    left = int(centerX - (width / 2))
                    top = int(centerY - (height / 2))

                    boxes.append([left, top, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)
            cv2.putText(frame, classes[class_ids[i]], (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
