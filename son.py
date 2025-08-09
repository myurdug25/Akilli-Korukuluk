import cv2
from ultralytics import YOLO
import threading
import pyrebase
import pygame
import time

firebaseConfig = {
  "apiKey": "AIzaSyAH9QDeESV4ZbFuyUyGT17lY5m0ICJ_3EE",
  "authDomain": "akillikorkuluk.firebaseapp.com",
    "databaseURL": "https://akillikorkuluk-default-rtdb.europe-west1.firebasedatabase.app/",
  "projectId": "akillikorkuluk",
  "storageBucket": "akillikorkuluk.firebasestorage.app",
  "messagingSenderId": "484162065521",
  "appId": "1:484162065521:web:9d4b3520ca08f7bf2577d0"
}
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# 📌 Ses yerine yazı bastır
def play_sound(animal_name):
    print(f"[SES] çalıştı: {animal_name}")

# 🔐 Giriş fonksiyonu
def login():
    print("🔐 Kullanıcı Girişi")
    email = input("E-posta: ")
    password = input("Şifre: ")
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        print("✅ Giriş başarılı.")
        return user['localId']
    except:
        print("❌ Giriş başarısız.")
        return None

# 📷 Kamera Thread
class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame
            else:
                print("❌ Kamera görüntüsü alınamadı.")
                self.running = False

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# 🐾 Hayvan isimleri ve varsayılan tepkiler
animal_names = ["kopek", "ayi", "domuz", "kus", "sansar", "kurt", "tilki"]
animal_sounds = {name: f"{name}.mp3" for name in animal_names}  # sadece gösterim için

# 🔁 Kullanıcı girişi
uid = None
while not uid:
    uid = login()

# 🔁 Kullanıcıya ait Realtime DB yapısını hazırla (varsa güncellemez)
for animal in animal_names:
    db.child("users").child(uid).child("detections").child(animal).set({
        "ses": 0,
        "isik": 0,
        "koku": 0
    })

# 🔍 YOLO modeli
model = YOLO("best.pt")

# 📷 Kamera başlat
video = VideoCaptureThread(0)  # PC kamerası

# 🎯 Algılama döngüsü
while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=320, conf=0.3)
    annotated_frame = results[0].plot()

    # Algılanan hayvanları al
    detected_animals = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

    for animal in detected_animals:
        if animal in animal_names:
            print(f"🚨 {animal.upper()} algılandı!")

            # 🔊 Ses (yazı ile gösterim)
            play_sound(animal)

            # 📡 Firebase güncelle
            db.child("users").child(uid).child("detections").child(animal).update({
                "ses": 1,
                "isik": 1,
                "koku": 1
            })

            # 2 saniye sonra resetle
            time.sleep(2)
            db.child("users").child(uid).child("detections").child(animal).update({
                "ses": 0,
                "isik": 0,
                "koku": 0
            })

    # 🎥 Gösterim
    cv2.imshow("YOLO Canlı Algılama", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🧹 Temizlik
video.stop()
cv2.destroyAllWindows()