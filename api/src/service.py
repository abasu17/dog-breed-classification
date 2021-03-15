import base64
import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image


class Prediction:
    
    DENSE_ACTIVATION = "softmax"
    CLASSES = None

    selected_labels = [
        "beagle", 
        "chihuahua", 
        "doberman", 
        "french_bulldog", 
        "golden_retriever", 
        "malamute", 
        "pug", 
        "saint_bernard", 
        "scottish_deerhound", 
        "tibetan_mastiff"]


    def __init__(self):

        self.CLASSES = len(self.selected_labels)

        global model
        model = Sequential()
        model.add(ResNet50(include_top = True, pooling = "avg", weights = "imagenet"))
        model.add(Dense(self.CLASSES, activation = self.DENSE_ACTIVATION))
        model.layers[0].trainable = False
        model.load_weights("./src/model/new_model.hdf5")


    def save_image(self, image):

        # store data as temp.jpg
        with open("temp.jpg", "wb") as fp:

            # write data into a file
            fp.write(base64.b64decode(image))


    def predict_image(self):
        original = image.load_img("temp.jpg", target_size=(224, 224))
        numpy_image = image.img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = preprocess_input(image_batch, mode='tf')
        preds = model.predict(processed_image)
        breed, score = max(zip(self.selected_labels, preds[0]), key=lambda x: x[-1])

        return {"breed" : str(breed), "score" : str(score) }