import cv2
import numpy as np 
import os

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