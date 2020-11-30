
from cv2 import cv2
import tensorflow as tf
import numpy as np
CATEGORIES =['cat','dog']
def prepare(filepath):
    IMG_SIZE=100
    img_array=cv2.imread(filepath)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(1,IMG_SIZE,IMG_SIZE,3)

    


model=tf.keras.models.load_model("mymodel2.tf")
prediction=model.predict([prepare('dogtest.jpg')])
if prediction[0][0]>prediction[0][1]:
    print('cat')
else:
    print('dog')
print(prediction[0][0],prediction[0][1])
