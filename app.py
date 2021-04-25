from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'CNN_MODEL.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(120, 160, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image__mean = np.mean(test_image)
    test_image_std = np.std(test_image)
    test_image = (test_image - test_image__mean)/test_image_std
    preds = model.predict(test_image)
    pred_class = model.predict_classes(test_image)
    return preds, pred_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds, pred_class = model_predict(file_path, model)
        label_list = ['akiec', 'bcc', 'bkl', 'df',  'nv', 'mel', 'vasc']
        l = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
             'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

        result = l[preds.argmax()]
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # Convert to string
        return result
    return "hello"


if __name__ == '__main__':
    app.run(debug=True)
