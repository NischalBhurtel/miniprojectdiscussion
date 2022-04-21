import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
img_dims=(96,96,3)
import numpy as np
model=load_model('ocrmodel.model')
img = cv2.imread(r'C:\Users\Acer\Desktop\copy.png')
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


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

