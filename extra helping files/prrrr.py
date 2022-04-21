from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D, Activation,Flatten,Dropout,Dense
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
epochs=1
lr=1e-3
batch_size=64
img_dims=(96,96,3)
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
data=[]
labels=[]
image_files=[f for f in glob.glob(r'C:\Users\Acer\Desktop\miniprojectdiscussion\Dataset'+"/**/*",recursive=True)if not os.path.isdir(f)]
random.shuffle(image_files)
for img in image_files:
  image=cv2.imread(img)
  image=cv2.resize(image,(img_dims[0],img_dims[1]))
  image=img_to_array(image)
  data.append(image)
  label=img.split(os.path.sep)[-2]
  if label=="A":
    label=0
  elif label=="B":
    label =1
  elif label=="C":
      label=2  
  elif label=="D":
      label=3
  elif label=="E":
      label=4   
  elif label=="F":
      label=5
  elif label=="G":
    label=6
  elif label=="H":
    label=7
  elif label=="I":
     label=8
  elif label=="J":
    label=9
  elif label=="K":
    label=10
  elif label=="L":
    label=11
  elif label=="M":
    label=12
  elif label=="N":
    label=13
  elif label=="O":
    label=14
  elif label=="P":
    label=15
  elif label=="Q":
    label=16
  elif label=="R":
    label=17
  elif label=="S":
    label=18
  elif label=="T":
    label=19
  elif label=="U":
    label=20
  elif label=="V":
    label=21
  elif label=="W":
    label=22
  elif label=="X":
    label=23
  elif label=="Y":
    label=24
  elif label=="Z":   
      label=25                                 
  labels.append([label])
data=np.array(data,dtype="float")/255.0
labels=np.array(labels)
(trainX, testX, trainY, testY)=train_test_split(data,labels,test_size=0.2,random_state=42) 
#trainX=trainX.reshape((-1,96,96,3))
#testX=testX.reshape((-1,96,96,3))
trainY=to_categorical(trainY,num_classes=26)  
testY=to_categorical(testY,num_classes=26)
print(np.shape(trainX))
print(np.shape(trainY))
aug=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
def build(width,height,depth,classes):
    model=Sequential()
    inputShape=(height,width,depth)
    chanDim=-1

    if K.image_data_format()=="channels_first":
        inputShape=(depth,height,width)
        chanDim=1

    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))    

    
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
   
   
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    
    
    model.add(Conv2D(128,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25)) 

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))   

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))
    
    return model
model=build(width=img_dims[0],height=img_dims[1],depth=img_dims[2],classes=26)
opt=Adam(lr=lr,decay=lr/epochs)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
model.fit_generator(aug.flow(trainX,trainY,batch_size=batch_size),validation_data=(testX,testY),steps_per_epoch=len(trainX) // batch_size,epochs=epochs,verbose=1)
model.save('ocrmodel')
#singletomultipleimageofadocument
img = cv2.imread(r'C:\Users\Acer\Desktop\copy.png')

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',img)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
												cv2.CHAIN_APPROX_NONE)



# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
#for cnt in contours:
cnt=contours[1]
	
	#if cnt==1:
x, y, w, h = cv2.boundingRect(cnt)
	
	  # Drawing a rectangle on copied image
	  #rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
	
	 # Cropping the text block for giving input to OCR
im2=img.copy()
cropped = im2[y:y + h, x:x + w] 
cv2.imshow('cropped',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


img=cv2.resize(im2,(img_dims[0],img_dims[1]))
img=img.astype("float")/255.0
img=img_to_array(img)
img=np.expand_dims(img,axis=0)

print(np.shape(img))

print(model.predict(img))
conf=model.predict(img)[0]
print(conf)

idx=np.argmax(conf)
label=classes[idx]
print(label)

