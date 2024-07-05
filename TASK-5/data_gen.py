import cv2
import numpy as np 
import os

# Part 1 - Dataset generation

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) 
cap.set(3, 1080)
cap.set(4, 640)

init = 1
  
while(True): 
    ret, image = cap.read() 
    faces = face_detector.detectMultiScale(image, 1.2, 5)

    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped_image = gray[y : y+h, x : x+w]
        cv2.imwrite(f'dataset/Hari_{init}.jpg', cropped_image)
        init += 1

    cv2.imshow('Image', image) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 

# Part 2 - Converting images to NumPy Object

def convert_images_to_npy():
    known_faces = []

    for file in sorted(os.listdir('dataset/')):
        img = cv2.imread(os.path.join('dataset',file), cv2.IMREAD_GRAYSCALE)  
        known_faces.append(cv2.resize(img, (100, 100)).reshape(1, -1))

    known_faces_array = np.array(known_faces)

    np.save('known_faces.npy',known_faces_array)

convert_images_to_npy()