import numpy as np
import cv2 as cv
import os

haar_cascade = cv.CascadeClassifier("src/haar_face.xml")

people_list = []
for people in os.listdir("images\Faces"):
    people_list.append(people)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("src/face_trained.yml")

img = cv.imread(r"images/Faces/Ben Afflek/9.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4) #Detect all faces
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people_list[label]} with a confidence of {confidence}')

    cv.putText(img, str(people_list[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0, 255), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), thickness=2)

cv.imshow("Detected Face", img)

cv.waitKey(0)