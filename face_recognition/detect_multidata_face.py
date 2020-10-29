import cv2
import numpy as np 
import os
import face_recognition
import os
from datetime import datetime

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
        # curImg = cv2.resize(curImg,(128,128))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
    print(myPicList)
# print(len(images))
print(" ")
# images = np.array(images)
# classNo = np.array(classNo)
# print(images.shape)
# print(classNo.shape)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(matches)
        # print(faceDis)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)
        if matches[matchIndex]:
        #     # name = classNames[matchIndex].upper()
            name = classNo
        #     #print(name)
        #     y1,x2,y2,x1 = faceLoc
        #     y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #     cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        #     cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        #     markAttendance(name)
        # else:
        # name =classNo[matchIndex]
        # y1,x2,y2,x1 = faceLoc
        # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        # cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        # cv2.putText(img,"name",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
 
    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) == 27:
        break  