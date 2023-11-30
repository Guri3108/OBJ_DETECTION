pip install opencv-python

import cv2

import matplotlib.pyplot as plt

config_file ="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

forzen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(forzen_model,config_file)

classLabels = []

file_name = 'labels.txt'

with open(file_name,'rt') as fpt:

    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)

print(len(classLabels))

model.setInputSize(320,320)#size of new frame

model.setInputScale(1.0/127.5)#scale factor for frame(multiplier)

model.setInputMean((127.5,127.5,127.5))#set mean value for  cam

model.setInputSwapRB(True)

img=cv2.imread('boy.jpg')

plt.imshow(img)

ClassIndex, confidence, bbox= model.detect(img, confThreshold= 0.5)

print(ClassIndex)

font_scale=3

font= cv2.FONT_HERSHEY_PLAIN

for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):

    cv2.rectangle(img, boxes,(225,0,0),2)
    
    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#VIDEO

cap = cv2.VideoCapture("pexels-george-morina-5688492.mp4")

if not cap.isOpened():

    cap=cv2.VideoCapture(0)

if not cap.isOpened():

    raise IOError("Cant open the video")
    

font_scale=3

font = cv2.FONT_HERSHEY_PLAIN

while True:

    ret, frame =cap.read()
    
    ClassIndex, confidece, bbox =model.detect(frame, confThreshold=0.55)
    
    print(ClassIndex)
    
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame, boxes,(225,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)
                
    
    cv2.imshow("objdetection by simplilearn",frame)
    
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
        
cap.release()


cv2.destroyaLLWindows()



#CAMERA


cap = cv2.VideoCapture(1)

if not cap.isOpened():

    cap=cv2.VideoCapture(0)

if not cap.isOpened():

    raise IOError("Cant open the video")
    

font_scale=3

font = cv2.FONT_HERSHEY_PLAIN


while True:

    ret, frame =cap.read()
    
    ClassIndex, confidece, bbox =model.detect(frame, confThreshold=0.55)
    
    print(ClassIndex)
    
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame, boxes,(225,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=3)
                
    
    cv2.imshow("objdetection by simplilearn",frame)
    
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
        

cap.release()

cv2.destroyaLLWindows()
