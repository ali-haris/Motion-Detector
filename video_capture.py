import cv2 ,time
import pandas as pd
from datetime import datetime as dt

video = cv2.VideoCapture(1)
first_fram = None
df = pd.DataFrame(columns=['start','end'])

status_list = [None,None]
time_list = []
while True:
    status = 0
    check , frame = video.read()
    # print(check)
    # print(frame)


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_fram is None:
        first_fram = gray
        continue


    delta_fram = cv2.absdiff(first_fram,gray)
    treshold_fram = cv2.threshold(delta_fram,30,255,cv2.THRESH_BINARY )[1]
    treshold_fram = cv2.dilate(treshold_fram,None,iterations=2)


    (cnts,_) = cv2.findContours(treshold_fram.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2 )


    if status_list[-1] == 1 and status_list[-2] == 0 :
        time_list.append(dt.now())

    if status_list[-1] == 0 and status_list[-2] == 1 :
        time_list.append(dt.now())


    cv2.imshow('Capture',gray)
    cv2.imshow('Blure', delta_fram)
    cv2.imshow('threshold',treshold_fram)
    cv2.imshow('color fram',frame)

    key =cv2.waitKey(1)
    if key == ord('q'):
        break
    # print(delta_fram)

    
    # print(cnts)
    print(status)
    status_list.append(status)

print(status_list)
print(time_list)

for i in range(0,len(time_list),2):
    df = df.append({'start':time_list[i],'end':time_list[i+1]},ignore_index=True)

df.to_csv('motion.csv')

video.release()

cv2.destroyAllWindows()
