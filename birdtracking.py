import cv2
import numpy as np
from ultralytics import YOLO

# Initialize webcam
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("Camera couldn't Access!!!")
    exit()

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'bird' is in COCO dataset

while True:
    success, img = cap.read()
    if not success:
        break

    # Run detection
    results = model(img, stream=True)

    bird_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "bird":
                bird_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Map to servo angles
                servoX = np.interp(cx, [0, ws], [180, 0])
                servoY = np.interp(cy, [0, hs], [180, 0])
                servoX = np.clip(servoX, 0, 180)
                servoY = np.clip(servoY, 0, 180)

                # Draw tracking UI
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(img, (cx, cy), 80, (0, 255, 0), 2)
                cv2.putText(img, f"({cx}, {cy})", (cx+15, cy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.line(img, (0, cy), (ws, cy), (0, 0, 0), 2)
                cv2.line(img, (cx, hs), (cx, 0), (0, 0, 0), 2)
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                break  # Only target first detected bird

    if not bird_detected:
        # No bird found
        cv2.putText(img, "NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (ws, 360), (0, 0, 0), 2)
        cv2.line(img, (640, hs), (640, 0), (0, 0, 0), 2)

    # Show video
    cv2.imshow("Bird Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
