import os
import logging

from flask import Flask, request, jsonify

from model.model import CatDogClassifier

app = Flask(__name__)  

# define model path
model_path = './model/final_model.h5'

# create instance
model = CatDogClassifier(model_path)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello world!"


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    image_url = request.args.get("image_url")
    prediction = model.predict(image_url)
    
    logging.info("prediction from model= {}".format(prediction))
    return jsonify({"predicted_class": str(prediction)})

def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False) 


if __name__ == "__main__":
    main()
