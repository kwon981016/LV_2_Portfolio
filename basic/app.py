
# some utilities
import os
import io
import numpy as np
from util import base64_to_pil
import base64

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, make_response, send_file

#tensorflow
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.rcParams["font.family"] = "NanumGothic"


# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
Model_json = "C:\\Users\\82104\\Downloads\\Flask (1)\\basic\\model\\json.json"
Model_weigths = "C:\\Users\\82104\\Downloads\\Flask (1)\\basic\\model\\model1.h5"


# Declare a flask app
app = Flask(__name__)


def get_ImageClassifierModel():
    # model = MobileNetV2(weights='imagenet')

    # Loading the pretrained model
    model_json = open(Model_json, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(Model_weigths)

    return model  
    


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    img = img.resize((180, 180))

    # Preprocessing the image
    x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = tf.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def main():
    cookie_name = 'test'
    resp = make_response(render_template('main.html'))
    resp.set_cookie('test', cookie_name, samesite='None', secure=True)
    return resp
    
   # return render_template('main.html')

@app.route("/photo")
def photo():
    return render_template("photo.html")    

@app.route("/eyewear")
def eyewear():
    return render_template("eyewear.html")

@app.route("/angulate")
def angulate():
    return render_template("angulate.html")

@app.route("/circle")
def circle():
    return render_template("circle.html")
    
@app.route("/long")
def long():
    return render_template("long.html")

@app.route("/triangle")
def triangle():
    return render_template("triangle.html")

@app.route("/egg")
def egg():
    return render_template("egg.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        global preds
        preds = model_predict(img, model)

        pred_proba = "{:.3f}".format(np.argmax(preds))    # Max probability
        
        global pred_class
        pred_class = ['각진 얼굴형','둥근 얼굴형','계란 얼굴형','긴 얼굴형','역삼각형 얼굴형']
        # pred_class = pred_class.decode('cp949').encode('utf-8')
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(pred_class[np.argmax(preds[0])], 100 * np.max(preds[0]))
        )
        
        result1 = np.argmax(preds[0])
        result1 = pred_class[result1]
        result2 = "{:.2f}".format(100 * np.max(preds[0]))
        

        result = f"{result2}%로, \n {result1}입니다."
        
        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)
    return None

@app.route('/print-plot')
def plot_png():
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        xs = pred_class
        ys = preds[0]*100
        axis.bar(xs, ys)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)