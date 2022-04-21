from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
img=cv2.imread( r'C:\Users\Acer\Desktop\miniprojectdiscussion\A.png')
img=cv2.resize(img,(96,96))
img=img_to_array(img)
img=np.expand_dims(img,axis=0)
print(np.shape(img))
cv2.imshow('image',img)
cv2.waitKey(2000)
cv2.destroyAllWindows()
