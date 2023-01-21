import cv2
import matplotlib.pyplot as plt
import numpy as np

plate_cascade = cv2.CascadeClassifier("/Users/eseosa/Desktop/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_russian_plate_number.xml")

number_plate = cv2.imread("/Users/eseosa/Desktop/car.png")

number_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2RGB)
number_plate_copy = number_plate.copy()
roi = number_plate.copy()

plate_rect = plate_cascade.detectMultiScale(number_plate_copy,
                                            scaleFactor=1.5,
                                            minNeighbors=5)

for (x, y, w, h) in plate_rect:

    #Detech number plate and draw rectangle

    cv2.rectangle(number_plate_copy, (x, y), (x+w, y+h), (255, 150, 50), 3)

    # MASK NUMBER PLATE ALL GRAY
    roi_sub = roi[y:y+h, x:x+w]

    roi_sub = np.ones(roi_sub.shape)*127

    number_plate[ y:y+h, x:x+w] = roi_sub

plt.imshow(number_plate)
plt.show()