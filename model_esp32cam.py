import cv2
import time
from ultralytics import YOLO

STREAM_URL = 'http://yourespip/stream'
model = YOLO('yolov8n.pt')

class_names = model.model.names

def main():
    print(f"Connecting to {STREAM_URL}...")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error, cannot open the stream video")
        print("Ensure that the IP address is correct and your ESP32-CAM is connected to WiFi.")
        return

    print("Connected! Press 'q' to exit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Stream disconected, trying to connect again")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue
        results = model(frame, stream=True, verbose=False)

        manusia_terdeteksi = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    manusia_terdeteksi = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Human {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if manusia_terdeteksi:
            print(f"Human detected")
        cv2.imshow("Stream Human Detection YOLOv8 (Push 'q' button to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream is closed")

if __name__ == "__main__":
    main()