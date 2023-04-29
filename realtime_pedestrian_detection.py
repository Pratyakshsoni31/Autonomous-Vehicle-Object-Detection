import cv2
import numpy as np

net = cv2.dnn.readNet("C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.weights", 
                      "C:\\Users\\Pratyaksh\\Downloads\\yolo\\yolov4-tiny.cfg")
classes = []
with open("C:\\Users\\Pratyaksh\\Downloads\\yolo\\\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = [net.getLayerNames()[i-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

confidence_threshold = 0.5
nms_threshold = 0.4

while True:
    ret, frame = cap.read()

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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 5), font, 1, color, 1)

    cv2.imshow("Pedestrian Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
