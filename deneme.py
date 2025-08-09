import cv2
from ultralytics import YOLO

rtsp_url = 'rtsp://admin:admin2024@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# YOLO modelini yükle
model = YOLO('yolov5n6.pt')

# Kamera açıldı mı kontrol et
if not cap.isOpened():
    print("Kameraya bağlanılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı!")
        break

    # YOLO ile nesne algılama
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Çizimli görüntüyü al
    annotated_frame = results[0].plot()

    # Ekranda göster
    cv2.imshow('YOLO IP Kamera', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
