from flask import Flask, render_template, request
from deepface import DeepFace
from PIL import Image
import numpy as np
from io import BytesIO


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    img1 = request.files['img1'].read()
    img2 = request.files['img2'].read()
    try:
        img1 = Image.open(BytesIO(img1))
        img1_np = np.array(img1)
        img2 = Image.open(BytesIO(img2))
        img2_np = np.array(img2)
        result = DeepFace.verify(img1_np, img2_np)
        return render_template('result.html', result=result)
    except Exception as e:
        print(str(e))
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
