import io
import json

from mmdet.apis import init_detector, inference_detector
import mmcv

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os 

app = Flask(__name__)
# class_index = json.load(open('class.json'))

config_file = './pretrained/CenterNet.py'
checkpoint_file = './pretrained/CenterNet.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cpu')

model.CLASSES = ("Amlodipine", "Simvastatin", "Losartan", "HCTZ", "Aspirin", "Metformin", "Enalapril",
        "VitaminB-Complex", "Glipizide", "Paracetamol", "CPM", "Diclofenac", "Ibuprofen",
        "Cetirizine", "Domperidone", "Amoxicillin", "Dicloxacillin")

class_list = ["Amlodipine", "Simvastatin", "Losartan", "HCTZ", "Aspirin", "Metformin", "Enalapril",
        "VitaminB-Complex", "Glipizide", "Paracetamol", "CPM", "Diclofenac", "Ibuprofen",
        "Cetirizine", "Domperidone", "Amoxicillin", "Dicloxacillin"]

cors = CORS(app, resources={r"/": {"origins": ""}})

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@cross_origin()
def home():
    main = 'Welcome_To_Drugtionary'
    return main

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(500),
        # transforms.CenterCrop(400),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    output = my_transforms(image).unsqueeze(0)
    return output

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        imgFile = f'uploads/{file.filename}'
        result = inference_detector(model, imgFile)

        drugdata = { 'classes': [], 'scores': [] }

        count = -1
        for drug in result:
            count += 1
            for pred_box in drug:
                if pred_box[-1] > 0.5:
                    drugdata["classes"].append(class_list[count])
                    drugdata["scores"].append(pred_box[-1])
        os.remove(imgFile)
        return {
            'classes': drugdata['classes'], 
            'scores': [str(_) for _ in drugdata['scores']]
            }

if __name__ == '__main__':
    app.run(debug=True, host='localhost')
