import tkinter as tk
from PIL import Image, ImageTk
import cv2
import imutils
from numpy import imag
# import videodet as vd

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
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # inisiasi list wajah, lokasi dan prediksi dari deteksi wajah
    faces = []
    locs = []
    preds = []

    # loop over the detections
    # Perulangan untuk deteksi
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
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
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return loakasi wajah dan hasil prediksi
    return (locs, preds)
    

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

def deteksi(frame):
    # Perulangan frmae videostream
    global stat
    while stat:
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
            frame = cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
        if stat == False:
            break

        return frame
# GUI
frame = tk.Tk()
frame.geometry("1000x680+200+10")
frame.title("Face Mask Detection by Kelompok 2 KB-L2")

# bgImage = tk.PhotoImage(file="bg.png")

bgImage = tk.PhotoImage(file="BG.png")

bg = tk.Label(frame, image=bgImage).place(x=0, y=0, relwidth=1, relheight=1)

vd = tk.Label(frame, bg="black")

def videostream():
    vd.place(x=180, y=112)   
    global vs, stat
    # Inisiasi Videostream 
    print("[INFO] starting video stream...")
    stat = True
    vs = VideoStream(src=0).start()
    # vs = cv2.VideoCapture(0)

    framevideo()


def framevideo():
    global stat
    frame = vs.read()
    frame = imutils.resize(frame, width = 640)
    frame = deteksi(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img  = Image.fromarray(frame)
    image = ImageTk.PhotoImage(image=img)
    vd.configure(image=image)
    vd.image = image
    vd.after(10, framevideo)


def stop() :
    global video, stat
    stat = False
    vd.place_forget()
    cv2.destroyAllWindows()
    vs.stop()


BStart = tk.Button(frame, text ="start", command = videostream, width=25, height=1, relief="flat")
BStart.place(x=290, y=620)
# BStart.pack()

BQuit = tk.Button(frame, text ="Stop", command = stop, width=25, height=1, relief="flat")
BQuit.place(x=490, y=620)
# BQuit.pack()

frame.mainloop()