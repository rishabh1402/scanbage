import os
import uuid
import flask
import torchvision.transforms as transforms
import urllib
from PIL import Image
import json
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from utils import load_classes
import io
from torchvision import models
import numpy as np
from flask import Flask, jsonify, request, render_template, redirect
from keras.applications.mobilenet import decode_predictions, preprocess_input
import tensorflow as tf


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
waste_model = load_model(os.path.join(BASE_DIR , 'model.hdf5'))
class_dict = load_classes('static/garbages/')
model = models.densenet121(pretrained=True)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model.eval() 

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    img = img_to_array(img)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = waste_model.predict(img)

    dict_result = {}
    for i in range(6):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:1]
    
    prob_result = []
    class_result = []
    for i in range(1):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

def garbage_class(class_name):
    """
    Classify an ImageNet category into 4 garbage classes and get CO2 mass.
    Example 'plastic_bag' -> 'recycle', '0.4'
    :param class_name: ImageNet class name.
    :return:
        Garbage class message.
        CO2 mass, also string.
    """
    normalized = class_name.split(',')[0].lower()
    for c, v in class_dict.items():
        for o, co in v:
            if normalized == o:
                message = c
                if c == 'compost':
                    message += ', check if still edible first'
                return message, co
    return "No garbage, detected a {}".format(normalized.replace('_', ' ')), "?"


@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = "static/images/upload.jpg"
    if request.method == 'POST':            
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                img_bytes = file.read()
                os.path.isfile(target_img)
                with open(target_img,"wb") as f:
                    f.write(img_bytes)
                #ImageNet prediction
                class_id, class_name = get_prediction(image_bytes=img_bytes)
                garbage_type, co = garbage_class(class_name)

                class_result , prob_result = predict(target_img , model)

                predictions = {
                      "class1":class_result[0],
                        "prob1": prob_result[0],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' ,
                                        predictions = predictions, 
                                        class_name=class_name, 
                                        garbage_type=garbage_type,
                                        co=co,
                                        )
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)
