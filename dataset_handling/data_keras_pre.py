import cv2
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

path = 'db'

images = []
classNo = [] 
myList = os.listdir(path)
print("Total number of peoples",len(myList))
noOfclasses = len(myList)
print("importing classes ")
for x in range(0,noOfclasses):
    myPicList = os.listdir(path+"/"+str(x))
    # print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
    print(myPicList)
# print(len(images))
print(" ")
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)

### spliting data
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=0.2)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.1)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

#checking train image
print(np.where(y_train==0))
print(len(np.where(y_train==0)[0]))


# for x in range (0,noOfclasses):
#     print(len(np.where(y_train==0)[0]))
noOfSamples =[]
for x in range (0,noOfclasses):
    noOfSamples.append(len(np.where(y_train==0)[0]))
print(noOfSamples)

#display

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfclasses),noOfSamples)
plt.title("no of images of each class")
plt.xlabel("class Id")
plt.ylabel("no of images")
plt.show()


#preprocessing

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

x_train = np.array(list(map(preprocessing,x_train)))
x_test = np.array(list(map(preprocessing,x_test)))
x_validation = np.array(list(map(preprocessing,x_validation)))

### reshape
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
x_train = x_train.shape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.shape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.shape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

########### keras pre the image #######
dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(x_train)
y_train = to_categorical(y_train,noOfclasses)
y_test = to_categorical(y_test,noOfclasses)
y_validation = to_categorical(y_validation,noOfclasses)

# https://www.youtube.com/watch?v=y1ZrOs9s2QA