# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect

from markupsafe import Markup
import numpy as np4
import openai
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')
openai.api_key = "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"
crop_info = {
    "rice": "A rice seed is the reproductive unit of the rice plant, containing the genetic information necessary for growth and development",
    "maize": "A maize seed is the embryo of a corn plant, containing the genetic information necessary for germination and growth.",
    "chickpea": "Chickpeas, also known as garbanzo beans, are nutritious legumes with a nutty flavor, high fiber, and protein.",
    "kidneybeans": "Kidney beans are a type of legume scientifically known as Phaseolus vulgaris. They are kidney-shaped.",
    "pigeonpeas": "Pigeonpea seeds are small, round legumes cultivated for their high protein content and resilience in diverse climates, serving as a staple food in many cultures worldwide.",
    "mothbeans": "Moth beans are small legumes prized for their high protein and fiber content. Cultivated in arid regions.",
    "mungbean": "Mung beans are small, green legumes native to Asia, prized for their nutty flavor and high nutritional value.",
    "blackgram": "Black gram, also known as urad dal, is a type of pulse widely used in Indian cuisine for its distinct flavor and high protein content.",
    "lentil": "Lentils are edible pulses renowned for their high protein and fiber content, making them a staple.",
    "pomegranate": "Pomegranate is a fruit known for its vibrant ruby-red arils packed with antioxidants, vitamins, and minerals.",
    "banana": "A banana seed is a small, black dot found within the fruit's flesh, responsible for propagating new banana plants when germinated.",
    "mango": "A coconut seed, encased in its hard shell, is the reproductive unit of the coconut palm tree, containing the embryo of a new plant ready for germination and growth.",
    "grapes": "A grape seed is a small, hard structure found within a grape, containing the genetic material necessary for plant reproduction.",
    "watermelon": "Watermelon seeds are small, edible seeds found within the flesh of a watermelon, containing essential nutrients like protein, healthy fats, and minerals.",
    "muskmelon": "A muskmelon seed is the small, oval-shaped reproductive unit found within the flesh of a muskmelon fruit.",
    "apple": "An apple seed is the small, embryonic plant enclosed in a protective outer covering found within the core of an apple fruit.",
    "orange": "An orange seed is the small, embryonic plant structure found within the pulp of an orange fruit, capable of germinating into a new orange tree.",
    "papaya": "Papaya seeds are small, black seeds found in the center of a papaya fruit. They are edible and have a slightly peppery flavor.",
    "coconut": "A coconut seed, encased in its hard shell, is the reproductive unit of the coconut palm tree.",
    "cotton": "A cotton seed is the small, oval-shaped seed of the cotton plant, containing fibers used in textile production and oil used in various industries.",
    "jute": "A jute seed is the reproductive unit of the jute plant, containing genetic material for growth. It develops into a fibrous plant used for making textiles.",
    "coffee": "A coffee seed is the small, bean-like structure found inside the red or purple fruit of the coffee plant.",
}
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu'), weights_only=True))
disease_model.eval()
# supplement_df = pd.read_csv("supplement_info.csv")
# supplements = supplement_df.to_dict(orient="records")




def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------




app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Smart Farm - Home'
    return render_template('index.html', title=title)









@app.route('/disease-predict', methods=['GET', 'POST'])
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'MyCrop - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            # img.save("send.jpg")

            prediction = predict_image(img)
            print(prediction)

            prediction_text = str(disease_dic[prediction])  # Convert prediction to string if not already
            cause_start = prediction_text.find("Cause of disease:")
            prevention_start = prediction_text.find("How to prevent/cure the disease")

            cause_of_disease = prediction_text[cause_start + len(
                "Cause of disease:"):prevention_start].strip()
            prevention_methods = prediction_text[prevention_start + len(
                "How to prevent/cure the disease"):].strip()

            return render_template('disease-result.html', prediction = prediction, cause_of_disease=cause_of_disease, prevention_methods=prevention_methods, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

@app.route('/contact')
def contact():
    return render_template('contact-us.html')



# Load supplement data from CSV
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.json.get("message")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}]
    )
    return jsonify({"reply": response["choices"][0]["message"]["content"]})


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5400)

