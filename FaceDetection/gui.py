import tkinter as tk
from PIL import Image, ImageTk
import cv2
import imutils
from numpy import imag

frame = tk.Tk()
frame.geometry("1000x680+200+10")
frame.title("Deteksi Masker")

bgImage = tk.PhotoImage(file="bg.png")

bg = tk.Label(frame, image=bgImage).place(x=0, y=0, relwidth=1, relheight=1)

frame.mainloop()
