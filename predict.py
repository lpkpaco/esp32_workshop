import cv2
from ultralytics import YOLO


ESP32_URL = ""
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.3
IMGSZ = 320,480
SKIP_FRAMES = 8

model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print("Check IP Address and connection")
    exit()

print("Video stream on. Press 'q' to quit.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame.")
        break

    frame_count += 1

    if frame_count % SKIP_FRAMES == 0:
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            device='cpu',
            verbose=True
        )

        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame


    cv2.imshow("ESP32-CAM - YOLOv8 Pose Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
