import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Loading YOLO model
model = YOLO('best.pt')

# Loading class list
with open("classes.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Function to calculate the Euclidean distance between two bounding box centers
def calculate_distance(box1, box2):
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2
    distance = np.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
    return distance

# Function to check if a bounding box is inside the ROI
def is_inside_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    return x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2

# Initialize video capture
cap = cv2.VideoCapture('test.mp4')

# Define the ROI (x1, y1, x2, y2) 
roi = (200, 100, 800, 400)  

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:  
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    # Draw the ROI on the frame
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (roi[0], roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Retrieve bounding boxes and class labels
    bboxes = []
    labels = []
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        # Append bounding box and label for collision detection
        bbox = [x1, y1, x2, y2]
        if is_inside_roi(bbox, roi): 
            bboxes.append(bbox)
            labels.append(c)

    # Collision Detection Logic
    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes):
            if i != j and labels[i] == "pedastrian" and labels[j] == "vehicle":
                distance = calculate_distance(bbox1, bbox2)

                # If the distance is below a threshold, mark it as a potential collision
                if distance < 100:  
                    center1 = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                    center2 = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                    cv2.line(frame, center1, center2, (0, 0, 255), 2)
                    cv2.putText(frame, "Warning!", center1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame with annotations
    cv2.imshow("ROI Collision Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()