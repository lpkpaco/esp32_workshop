import gradio as gr
import cv2
import time
import requests
import os
import numpy as np
from ultralytics import YOLO


ESP32_IP = ""
CAPTURE_URL = f"http://{ESP32_IP}/capture"
MODEL_PATH = "best.pt" 
CONF_THRESHOLD = 0.35
IMGSZ = 768
FALL_DURATION = 3
TIMEOUT = 6.0
SAVE_DIR = "captured_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
lay_count = 0
alert_triggered = False
total_alerts = 0
last_state = None
last_detected_time = 0.0
running = False

def capture_and_predict():
    global lay_count, alert_triggered, total_alerts, last_state, last_detected_time

    if not running:
        return None, lay_count, "Not Running", total_alerts, ""

    current_time = time.time()
    frame = None
    label = None
    best_box = None
    best_conf = 0.0
    error_msg = ""

    try:
        response = requests.get(CAPTURE_URL, timeout=5)
        if response.status_code != 200:
            error_msg = f"HTTP error: {response.status_code}"
            return None, lay_count, "Error", total_alerts, error_msg

        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            error_msg = "Failed to decode image"
            return None, lay_count, "Error", total_alerts, error_msg

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, device='cpu', verbose=False)
        best_box = None
        best_conf = 0.0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is not None and best_conf >= CONF_THRESHOLD:
            cls_id = int(best_box.cls[0])
            label = model.names[cls_id]   # 'stand', 'sit', or 'lay'
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            last_detected_time = current_time

            if label == 'lay':
                if last_state == 'stand':
                    lay_count = 1
                    alert_triggered = False
                elif last_state == 'lay':
                    lay_count += 1
                else: 
                    lay_count = 0
                    alert_triggered = False

                if lay_count >= FALL_DURATION and not alert_triggered:
                    alert_triggered = True
                    total_alerts += 1

                last_state = label

            else:
                if last_state == 'lay':
                    lay_count = 0
                    alert_triggered = False
                last_state = label

        else:
            if last_state is not None and (current_time - last_detected_time) > TIMEOUT:
                last_state = None
                lay_count = 0
                alert_triggered = False

        annotated = frame.copy()
        if best_box is not None and best_conf >= CONF_THRESHOLD:
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {best_conf:.2f}"
            cv2.putText(annotated, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if alert_triggered:
                cv2.putText(annotated, "!!! FALL DETECTED !!!",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 4)
                cv2.putText(annotated, f"Lay count: {lay_count}/{FALL_DURATION}",
                            (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 3)
        else:
            cv2.putText(annotated, "No person detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        status = "Alert Active" if alert_triggered else "Normal"
        return annotated, lay_count, status, total_alerts, error_msg

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, lay_count, "Error", total_alerts, error_msg

# Gradio UI
with gr.Blocks(title="Fall Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Fall Detection System")
    gr.Markdown("Monitor a single person for fall events (stand → lay continuously).")

    running_state = gr.State(value=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            start_btn = gr.Button("Start", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            save_checkbox = gr.Checkbox(label="Save images to disk", value=True)

            gr.Markdown("### Status")
            lay_count_display = gr.Number(label="Lay count (consecutive)", value=0, precision=0)
            alert_status_display = gr.Textbox(label="Alert status", value="Not Running", interactive=False)
            total_alerts_display = gr.Number(label="Total warnings triggered", value=0, precision=0)
            error_display = gr.Textbox(label="Error / Info", value="", interactive=False)

        with gr.Column(scale=2):
            img_display = gr.Image(label="Captured frame with inference", type="numpy")

    timer = gr.Timer(interval=3, active=False)

    def start():
        global running
        running = True
        return True, "Running", 0, 0, ""

    def stop():
        global running
        running = False
        return False, "Stopped", 0, 0, ""

    def timer_tick(running_flag, save_flag):
        global running
        if not running_flag:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        img, count, status, total, err = capture_and_predict()

        if img is not None and save_flag:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            cv2.imwrite(save_path, img)
        return img, count, status, total, err

    timer.tick(
        fn=timer_tick,
        inputs=[running_state, save_checkbox],
        outputs=[img_display, lay_count_display, alert_status_display, total_alerts_display, error_display]
    )

    start_btn.click(
        fn=lambda: (True, True),
        inputs=[],
        outputs=[running_state, timer]
    )


    def start_action():
        global running
        running = True
        return True, True

    def stop_action():
        global running
        running = False
        return False, False 

    start_btn.click(
        fn=start_action,
        inputs=[],
        outputs=[running_state, timer]
    )

    stop_btn.click(
        fn=stop_action,
        inputs=[],
        outputs=[running_state, timer]
    )

    stop_btn.click(
        fn=lambda: (None, "Stopped", 0, 0, ""),
        inputs=[],
        outputs=[img_display, alert_status_display, lay_count_display, total_alerts_display, error_display]
    )

if __name__ == "__main__":
    demo.launch()