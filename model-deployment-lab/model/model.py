import os
# hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from PIL import Image
from urllib.request import urlretrieve

import logging

class CatDogClassifier:

    def __init__(self, model_path):
        logging.info("CatDogClassifier class initialized")
        self.model = load_model(model_path)
        logging.info("Model is loaded!")
        

    def predict(self, image_url):
        # load the image
        img = self.load_image(image_url)

        # predict the class
        result = self.model.predict(img)

        # return class
        if result[0] == 0:
            return "Cat"
        else:
            return "Dog"

    # load and prepare the image
    def load_image(self, image_url):
        # download image
        img = self.download_url(image_url, "input.jpg")
        logging.info("Image downloaded from {}".format(image_url))

        # load the image
        img = load_img("input.jpg", target_size=(224, 224))
      
        # convert to array
        img = img_to_array(img)
      
        # reshape into a single sample with 3 channels
        img = img.reshape(1, 224, 224, 3)
      
        return img

    def download_url(self, url, filename):
        """Download a file from url to filename, with a progress bar."""
        urlretrieve(url, filename, data=None)

def main():
	model = CatDogClassifier('final_model.h5')
	predicted_class = model.predict("https://cdn.britannica.com/60/8160-050-08CCEABC/German-shepherd.jpg")
	logging.info("This is an image of a {}".format(predicted_class)) 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()