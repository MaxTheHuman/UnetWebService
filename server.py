import cv2
import io
import numpy as np

from PIL import ImageOps
from flask import Flask, request, send_file
from tensorflow import keras


app = Flask(__name__)

@app.route('/', methods=['GET'])
def main_page():
    source_code = open('index.html', 'r', encoding='utf-8').read()  
    return source_code
    

@app.route('/convert', methods=['POST'])
def convert():    
    # convert string of image data to uint8
    nparr = np.frombuffer(request.files['file'].read(), np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_size = (160, 160)
    resized_image = cv2.resize(img, img_size)

    # load pretrained model
    model = keras.models.load_model('/Users/m.siplivy/unet_service/trained_model')
    resized_image = np.reshape(resized_image, [1, 160, 160, 3])
    
    pred = model.predict(resized_image)
    
    mask = np.argmax(pred[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    b = io.BytesIO()
    mask.save(b, 'jpeg')
    im_bytes = b.getvalue()

    return send_file(
        io.BytesIO(im_bytes),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='img.jpeg')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
