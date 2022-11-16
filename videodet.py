# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# fungsi untuk mendeteksi lokasi wajah dan presiksi masker


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # inisiasi list wajah, lokasi dan prediksi dari deteksi wajah
    faces = []
    locs = []
    preds = []

    # Perulangan untuk deteksi
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # menambahkan wajah dan lokasi pada list
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # if untuk membuat deteksi ketika setidaknya ada satu wajah yang terdeteksi
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return loakasi wajah dan hasil prediksi
    return (locs, preds)


# load face detector
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load model
maskNet = load_model("mask_detector.model")

# Inisiasi Videostream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


# Perulangan frmae videostream
while True:
    # Mengambil frame dari thread videostream dan me resizenya menjadi 1000 px
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    # mendeteksi wajah dengan memanggil fungsi detect_and_predict_mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Perulangan untuk lokasi dan prediksi wajah
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Menentukan class label dan color yang akan digunakan untukk menampilkan warna box dan text label
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Menambahkan Probabilitas deteksi pada label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Memasukan text label serta box deteksi pada output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Memunculkan Frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if untuk break perulangan frame deteksi
    if (key == ord("q")) | (cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1):
        break


cv2.destroyAllWindows()
vs.stop()
