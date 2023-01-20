import cv2
import matplotlib.pyplot as plt

"""
We will use MatchTemplate algorithm to match a picture of a face to another picture of the same face but in a 
group of two people. We will start by uploading both pictures needed for the this experiment using the Opencv imread
method.
"""

full = cv2.imread("/Users/eseosa/Desktop/group_of_2.png")
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)

face = cv2.imread("/Users/eseosa/Desktop/face.png")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
plt.imshow(face)
plt.show()

# We will Evaluate the inference result of the 6 methods in the list below.

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for choice in methods:

    full_copy = full.copy()

    method = eval(choice)

    result = cv2.matchTemplate(full_copy, face, method)

    min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)

    height, width, channels = face.shape


    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_location  #(x, y)

        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(full_copy, top_left, bottom_right, (0, 0, 255), 10)

    else:
        top_left = max_location  # (x, y)

        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(full_copy, top_left, bottom_right, (0, 255, 255), 10)


    plt.imshow(full_copy)
    plt.title("face detection")

    plt.suptitle(choice)
    plt.show()

