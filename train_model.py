import pickle
import tensorflow as tf

x=pickle.load(open('X.pkl','rb'))
y=pickle.load(open('Y.pkl','rb'))
x=x/255
print(x.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model=Sequential()
model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation= 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape=x.shape[1:],activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=12,validation_split=0.1)

model.save('mymodel2.tf')
loaded_model = tf.keras.models.load_model('mymodel2.tf')
#print(model.predict(x))
