import time

from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageFilter
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from typing import List
import skimage.transform as trans
from warnings import filterwarnings
import time
from google.cloud import storage
from firebase import firebase
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

from collections import Counter

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
db_url = 'https://console.firebase.google.com/project/images-8647c'  # Your project url
firebase = firebase.FirebaseApplication(db_url, None)
client = storage.Client()
bucket = client.get_bucket('images-8647c.appspot.com')
imageBlob = bucket.blob("/")

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = ['glioma', 'meiningioma','notumor','pituitary']

model_type = load_model("model44.h5")

model = load_model("BT_Model.h5")



SegModel = load_model("seg.h5")



def Class_Pridict(path):

    def scalar(img):
        return img

    others_generator = ImageDataGenerator(preprocessing_function=scalar)

    data_paths_series = pd.Series(Path(path), name="Images").astype(str)
    data_labels_Series = pd.Series("pred", name="TUMOR_Category")
    Main2 = pd.concat([data_paths_series, data_labels_Series], axis=1)

    img_gen = others_generator.flow_from_dataframe(dataframe=Main2,x_col="Images",y_col="TUMOR_Category",color_mode="rgb",class_mode="categorical",shuffle=False,target_size=(256, 256))


    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = trans.resize(image, (256, 256, 1))

    p = model.predict(np.reshape(image, (1, 256, 256, 1)))




    predicted = model_type.predict(img_gen)

    if p[0] > 0.5:
        if np.argmax(predicted[0]) == 2:
            return "Unspecified type"
        else :
            return classes[np.argmax(predicted[0])] + "Case 1"
    else:
        if np.argmax(predicted[0]) == 2:
            return "notumor"
        else :
            return classes[np.argmax(predicted[0])] + "Case 2"




def SegPredict(path, color):
    r, g, b = color

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = trans.resize(image, (256, 256, 1))

    rows, cols, ch = image.shape
    predicted = SegModel.predict(np.reshape(image, (1, 256, 256, 1)))
    predicted = np.reshape(predicted, (256, 256))
    predicted = predicted.astype(np.float32) * 255
    predicted = trans.resize(predicted, (rows, cols, 1))
    predicted = predicted.astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)

    x = np.array(image)
    x = trans.resize(image, (256, 256, 3))
    y = np.array(predicted)

    ret, mask = cv2.threshold(y, 120, 255, cv2.THRESH_BINARY)
    white_pixels = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
    mask[white_pixels] = [r, g, b]
    mask = np.array(mask)
    s=abs(x - mask)
    s=np.array(s*255, dtype=np.uint8)
    img = Image.fromarray(s)
    img.save("img.jpeg")


@app.get('/')
async def home():
    return "Hello World"


@app.post('/multifiles')
async def Upload2(files : List[bytes] = File(...)):

    [f.unlink() for f in Path("Images").glob("*") if f.is_file()]

    x=[file for file in files]
    i=0
    for f in  x :
        with open(f'Images/{i}.jpg', 'wb') as image:
            image.write(f)
            image.close()
            i+=1


    return Class_multi_Pridict()

@app.post('/multiSeg')
async def SegMulti():

    return Seg_Multi_Predict()


@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open(f'as.jpg', 'wb') as image:
        image.write(file)
        image.close()

    filterwarnings("ignore", category=DeprecationWarning)
    filterwarnings("ignore", category=FutureWarning)
    filterwarnings("ignore", category=UserWarning)

    r= Class_Pridict('as.jpg')

    return r
@app.post("/seg")
async def UploadImage1(file: bytes = File(...)):
    with open(f'as.jpg', 'wb') as image:
        image.write(file)
        image.close()
        filterwarnings("ignore", category=DeprecationWarning)
        filterwarnings("ignore", category=FutureWarning)
        filterwarnings("ignore", category=UserWarning)

        SegPredict('as.jpg', (0, 255, 255))
        filename = str(int(time.time()))
        imagePath = "img.jpeg"

        imageBlob = bucket.blob(filename)
        imageBlob.upload_from_filename(imagePath)
        imageBlob.make_public()

        return imageBlob.public_url





 # Upload your image

def Class_multi_Pridict():

    def scalar(img):
        return img

    dataa_paths_list = list(Path("Images").glob("*.jpg"))

    data_paths_list = []
    data_labels_list = []

    for p in dataa_paths_list:
        data_paths_list.append(p)
        data_labels_list.append("pred")

    others_generator = ImageDataGenerator(preprocessing_function=scalar)

    data_paths_series = pd.Series(data_paths_list, name="Images").astype(str)
    data_labels_Series = pd.Series(data_labels_list, name="TUMOR_Category")
    Main2 = pd.concat([data_paths_series, data_labels_Series], axis=1)

    img_gen = others_generator.flow_from_dataframe(dataframe=Main2,x_col="Images",y_col="TUMOR_Category",color_mode="rgb",class_mode="categorical",shuffle=False,target_size=(256, 256))



    predicted = model_type.predict(img_gen)

    results = []

    for p in predicted:
        results.append(classes[np.argmax(p)])

    c = Counter(results)
    c.most_common(1)
    r = c.most_common()[0][0]

    return r


def Seg_Multi_Predict():

    Pathes = list(Path("Images").glob("*.jpg"))
    links = []

    for i in range(0,len(Pathes)) :

        SegPredict(f"Images/{i}.jpg", (0, 255, 255))

        filename = str(int(time.time()))
        imagePath = "img.jpeg"
        imageBlob = bucket.blob(filename)
        imageBlob.upload_from_filename(imagePath)
        imageBlob.make_public()

        links.append(imageBlob.public_url)

    return links

