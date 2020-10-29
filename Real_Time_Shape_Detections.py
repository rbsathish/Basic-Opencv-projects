import cv2
import numpy as np 


frameWidth = 320#640
frameHeight = 320#480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

# img = cv2.imread('shapes.jpg')
# cv2.imshow("original image",img)
# cv2.waitKey(0)
def empty(a):
    pass

cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",320,320)
cv2.createTrackbar("Threshold1","parameters",23,255,empty)
cv2.createTrackbar("Threshold2","parameters",20,255,empty)
cv2.createTrackbar("Area","parameters",500,30000,empty)

def stakImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0],list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(0,0),None,scale,scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(imgArray[0][0].shape[1],imgArray[0][0].shape[0]),None,scale,scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height,width,3),np.uint8)
        hor = [imageBlank]*rows
        hor_cols = [imageBlank]*cols
        for x in range(0,rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2]== imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x],(imgArray[0].shape[1],imgArray[0].shape[0]),None,scale,scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
        
def getContours(img,imgcontour):
    _,contours,hieracchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)               
    
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area","parameters")
        if area >areaMin:
            cv2.drawContours(imgcontour,cnt,-1,(255,0,255),7)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),5)
            cv2.putText(imgcontour,"Points : " + str(len(approx)),(x+w+20,y+20),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),2)
            cv2.putText(imgcontour,"Area : " + str(int(area)),(x+w+20,y+45),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),2)
            
        
    
while True:
    success, img = cap.read()
    imgcontour = img.copy()
    Blur = cv2.GaussianBlur(img,(7,7),1)
    Gray = cv2.cvtColor(Blur,cv2.COLOR_BGR2GRAY)
    
    threshold1 = cv2.getTrackbarPos("threshold1","parameters")
    threshold2 = cv2.getTrackbarPos("threshold2","parameters")
    Canny = cv2.Canny(Gray,threshold1,threshold2)
    
    kernel = np.ones((5,5))
    Dilate = cv2.dilate(Canny,kernel,iterations=1)
    
    getContours(Dilate,imgcontour)
    
    imageStack = stakImages(0.8,([img,Gray,Canny],[Dilate,imgcontour,imgcontour]))
    cv2.imshow("output",imageStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
    
    