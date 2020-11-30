import numpy as np
from cv2 import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

DIRECTORY=r'/home/android/Student/Swastik/cat-dog-classify/dataset'
CATEGORIES=['cats','dogs']
data=[]
for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    label=CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(100,100))
        data.append([img_arr,label])
random.shuffle(data)
X=[]
Y=[]
for features,labels in data:
    X.append(features)
    Y.append(labels)
X=np.array(X)
Y=np.array(Y)
print(X)
print(Y)
pickle.dump(X,open('X.pkl','wb'))
pickle.dump(Y,open('Y.pkl','wb'))
        
        
        
        
        
