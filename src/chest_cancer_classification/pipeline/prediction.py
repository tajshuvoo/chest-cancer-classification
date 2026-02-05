import numpy as np
import tensorflow as tf
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = tf.keras.models.load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = tf.keras.preprocessing.image.load_img(imagename, target_size = (224,224))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)

        probs = model.predict(test_image)
        result = np.argmax(probs, axis=1)
        class_map = {0: 'Adenocarcinoma Cancer', 1: 'Normal'}
        prediction = class_map.get(result[0], "Unknown")
        return [{
            "image": prediction,
            "probabilities": probs.tolist(),
            "predicted_class": int(result[0])
        }]