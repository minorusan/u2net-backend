import os
import gc
import io
import time
import base64
import logging

import numpy as np
from PIL import Image

from flask import Flask, request, send_file, jsonify, json
from flask_cors import CORS

import detect


app = Flask(__name__)
CORS(app)

net = detect.load_model(model_name="u2netp")

logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
def ping():
    return "U^2-Net!"


@app.route("/remove", methods=["POST"])
def remove():

    start = time.time()

    logging.info('got there!')
    
    json_object = json.loads(request.data)
    bytes = bytearray(json_object['file']['data'])
    
    img = Image.open(io.BytesIO(bytes))

    output = detect.predict(net, np.array(img))
    output = output.resize((img.size), resample=Image.BILINEAR) # remove resample

    empty_img = Image.new("RGBA", (img.size), 0)
    new_img = Image.composite(img, empty_img, output.convert("L"))

    buffer = io.BytesIO()
    new_img.save(buffer, "PNG")
    buffer.seek(0)

    logging.info(f" Predicted in {time.time() - start:.2f} sec")
    
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
