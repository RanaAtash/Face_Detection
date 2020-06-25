# import openCV library
import cv2

# Face detection in video stream using "Haar feature-based cascade classifiers"
# this method was proposed by Paul Viola and Michael Jones in their paper
# "Rapid Object Detection using a Boosted Cascade of Simple Features"

# import the face cascade classifier
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# open a live video stream 
cap = cv2.VideoCapture(0)

while True:
    # capture each frame
    ret, frame = cap.read()
    # convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    # draw a rectangle around any detected face inside the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # display the frame
    cv2.imshow('Frame', frame)
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()