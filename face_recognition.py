import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier("src/haar_face.xml")

people_list = []
for people in os.listdir("images\Faces"):
    people_list.append(people)

features = np.load("features.npy")
labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(r"images/Faces/Ben Afflek/1.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces_rect:
    faces_roi = [y:y+h, x:x+h]
