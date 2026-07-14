import cv2
import time
import requests
import os
from ultralytics import YOLO
ESP32_IP = ""
CAPTURE_URL = f"http://{ESP32_IP}/capture"
MODEL_PATH = "best.pt"
INTERVAL = 3.0
CONF_THRESHOLD = 0.35
SAVE_DIR = "captured_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
print(f"Taking photo wwith {INTERVAL} seconds interval.")

while True:
    try:
        start_time = time.time()
        response = requests.get(CAPTURE_URL, timeout=5)
        if response.status_code != 200:
            print(f"Error code: {response.status_code}")
            time.sleep(INTERVAL)
            continue

        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode image")
            time.sleep(INTERVAL)
            continue

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=768,1024, device='cpu', verbose=True)
        annotated_frame = results[0].plot()
        cv2.imshow("ESP32-CAM Prediction", annotated_frame)
        cv2.waitKey(1)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
        cv2.imwrite(save_path, annotated_frame)
        print(f"Prediction saved: {save_path}")

        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by user")
        break

cv2.destroyAllWindows()
