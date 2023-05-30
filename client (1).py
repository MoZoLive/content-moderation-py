import requests
import json
import cv2
import base64
import os

def videoframes(filename):
    vid = cv2.VideoCapture(filename)
    if not os.path.exists('images'):
        os.makedirs('images')
    index = 0
    while (True):
        ret, frame = vid.read()
        if not ret:
            break
        name = './images/frame' + str(index) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        index += 1

imgdata=[]
videodata=[]
def database(db):
    os.chdir(db)
    for current_dir, dirs, files in os.walk('.'):
        for img_video in files:
            if img_video.endswith('.jpg'):
                imgdata.append(img_video)
            elif img_video.endswith('.mp4'):
                videodata.append(img_video)
    return imgdata, videodata

url = 'http://192.168.31.200:8000/detect'
def sendrequest(filename):
    with open(filename, 'rb') as f:
        r = requests.post(url, json={'image': base64.b64encode(f.read()).decode()})
        print(r.text)
        return r.text

#main
img_output=[]
videoimg_output=[]
print("waiting for the response.....")
image_data, video_data=database(r"E:\pycharm projects\nuditydetection\database")
if len(image_data) is not None:
    for filename in image_data:
        output=sendrequest(filename)
        img_output.append(output)
    print("img_output", img_output)

if len(video_data) is not None:
    for filename in video_data:
        videoframes(filename)
        breakedframes, video =database(r'E:\pycharm projects\nuditydetection\database\images')
        for filename in breakedframes:
            output = sendrequest(filename)
            videoimg_output.append(output)
    print("videoimg_output", videoimg_output)

# OPEN THE EMULATOR AND PASTE THE FOLLOWING URL:
# http://192.168.1.104:8000/test
# if you get errors, it means emulator don't work and you need to debug on a physical phone
