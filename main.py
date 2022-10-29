from tokenize import Imagnumber
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import os, ssl
from PIL import Image
import PIL.ImageOps

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Fetching the data
X = np.load("image.npz")
y = pd.read_csv('mnist_784', version=1, return_X_y=True)

print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","Y","Z"]
nclasses = len(classes)

# Splitting the data

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state = 9, train_size = 7500, test_size = 2500)
x_trainscaled = x_train/255.0
x_testscaled = x_test/255.0
print(X.loc[0])
print(y.loc[0])


clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_trainscaled, y_train)

y_pred = clf.predict(x_testscaled)

accuracy = accuracy_score(y_test, y_pred)

print('The accuracy is: ', accuracy)


cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape()
        upper_left  = (int(width/2-56), int(height/2-56))
        bottom_right  = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0),2 )

        ROI = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]


        im_pil = Image.fromarray(ROI)

       
        imgbw = im_pil.convert('L')
        imgbwresized = imgbw.resize((28,28), Image.ANTIALIAS)

        imgbwresizedinverted = PIL.ImageOps.invert(imgbwresized)
        pixelfilter = 20
        minimumpixel = np.percentile(imgbwresizedinverted,pixelfilter)
        imgbwresizedinvertedscaled = np.clip(imgbwresizedinverted-minimumpixel,0,255)
        maximumpixel = np.max(imgbwresizedinverted)
        imgbwresizedinvertedscaled = np.asarray(imgbwresizedinvertedscaled)/maximumpixel
        testsample = np.array(imgbwresizedinvertedscaled.reshape(1,784))
        testpred = clf.predict(testsample)
        print("predicted class is:",testpred)

        #displaying the result frame
        cv2.imshow("frame",gray)
        if cv2.waitkey(1) & 0xFF == ord(q):
            break

    except Exception as e:
        pass

#releasing the windows and camera
cap.release()
cv2.DestroyAllWindows()


