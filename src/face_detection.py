from distutils.fancy_getopt import fancy_getopt
import cv2 as cv

img = cv.imread("images/4people.webp")

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Haar Cascade Classifier
haar_cascade = cv.CascadeClassifier("src/haar_face.xml")
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

cv.imshow("Detected Faces", img)

cv.waitKey(0)
