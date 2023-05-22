import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch
from torchvision import transforms
from PIL import Image
import json
from train import PreTrainedResNet

app = Flask(__name__)
model = torch.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/flask/model.pkl','rb'),map_location=torch.device('cpu'))
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# model = pickle.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():  
    # model = PreTrainedResNet()
    # model = model.load_state_dict(torch.load("C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/flask/model.pkl",'rb'),map_location=torch.device('cpu'))
    model = torch.load(open('C:/Users/umar  masood/Favorites/Downloads/Deployment-flask-master/Deployment-flask-master/flask/model.pkl','rb'),map_location=torch.device('cpu'))
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    if request.method =='POST':
       image_file =  request.files['image']
    #    print(image_file)
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load and preprocess the image
    image = Image.open(image_file)
    print(image)
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     image = image.cuda()

    # Pass the image through the model
    
    
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_index = predicted.item()
    class_label = class_names[class_index]
    return class_label


# def predict_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image found'})

#     image = request.files['image']
#     image_path = "path_to_save_uploaded_image.jpg"  # Provide a path to save the uploaded image
#     image.save(image_path)

#     predicted_class = predict(model, image_path, class_names)

#     return jsonify({'predicted_class': predicted_class})

   
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('new.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":

    app.run(debug=True)