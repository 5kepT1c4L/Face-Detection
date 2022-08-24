import cv2 as cv
import numpy as np
import os 

people_list = []
for people in os.listdir("images\Faces"):
    people_list.append(people)

DIR = "images\Faces"
haar_cascade = cv.CascadeClassifier("src/haar_face.xml")

features = []
labels = []
def create_train():
    for people in people_list:
        PATH = os.path.join(DIR, people) # Every individual's picture folder
        label = people_list.index(people)

        for img in os.listdir(PATH): # Every picture in each individual's picture folder
            img_path = os.path.join(PATH, img) 

            img_array = cv.imread(img_path) # Read every image
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)


