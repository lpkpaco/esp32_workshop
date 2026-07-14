import cv2
ESP32_URL = ""
cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print("Could not open the stream.")
    print("Check your IP address, ensure the ESP32 is powered, and verify it's connected to the same Wi-Fi network.")
    exit()
print("Stream successfully opened! Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame.")
        break
    cv2.imshow("ESP32-CAM Raw Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
