import cv2
import time
import requests
import os
import numpy as np
from ultralytics import YOLO

ESP32_IP = ""
CAPTURE_URL = f"http://{ESP32_IP}/capture"
MODEL_PATH = "best.pt"
INTERVAL = 3.0
CONF_THRESHOLD = 0.35
IMGSZ = 768
TIMEOUT = 6.0
SAVE_DIR = "captured_predictions"

os.makedirs(SAVE_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

last_state = None
lay_count = 0
alert_triggered = False
last_detected_time = 0.0

try:
    while True:
        start_loop = time.time()
        current_time = time.time()

        try:
            response = requests.get(CAPTURE_URL, timeout=5)
            if response.status_code != 200:
                print(f"HTTP Error: {response.status_code}")
                time.sleep(INTERVAL)
                continue

            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode image")
                time.sleep(INTERVAL)
                continue

        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")
            time.sleep(INTERVAL)
            continue

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMGSZ,
            device='cpu',
            verbose=False
        )

        best_box = None
        best_label = None
        best_conf = 0.0

        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is not None and best_conf >= CONF_THRESHOLD:
            cls_id = int(best_box.cls[0])
            label = model.names[cls_id]  # 'stand', 'sit', 'lay'
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            last_detected_time = current_time

            if label == 'lay':
                if last_state == 'stand':
                    lay_count = 1
                    alert_triggered = False
                    print(f"[Status] stand -> lay, counting (1/3)")
                elif last_state == 'lay':
                    lay_count += 1
                    print(f"[Status] kept laying, counting {lay_count}/3")
                else:
                    lay_count = 0
                    alert_triggered = False
                    print(f"[Status] {last_state} changed to lay, not counting")

                if lay_count >= 3 and not alert_triggered:
                    alert_triggered = True
                    print(f"Alarm triggered due to laying down for long time")

                last_state = label

            else:
                if last_state == 'lay':
                    print(f"[Status] Stopped laying, resetting count")
                lay_count = 0
                alert_triggered = False
                last_state = label

        else:
            if last_state is not None:
                if current_time - last_detected_time > TIMEOUT:
                    print(f"[Status] Resetting count due to {TIMEOUT} seconds not detecting anyone.")
                    last_state = None
                    lay_count = 0
                    alert_triggered = False

        annotated = frame.copy()
        if best_box is not None and best_conf >= CONF_THRESHOLD:
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{best_label} {best_conf:.2f}"
            cv2.putText(annotated, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if alert_triggered:
                cv2.putText(annotated, "!!! FALL DETECTED !!!",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 4)
                cv2.putText(annotated, f"Lay count: {lay_count}/3",
                            (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 3)
        else:
            cv2.putText(annotated, "No person detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ESP32-CAM Fall Detection", annotated)
        cv2.waitKey(1)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
        cv2.imwrite(save_path, annotated)
        print(f"Saved: {save_path}  (Status: {last_state}, Count: {lay_count}, Alarm: {alert_triggered})")

        elapsed = time.time() - start_loop
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nUser Interruption")

cv2.destroyAllWindows()
