import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from chest_cancer_classification.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Unfreeze last 4 layers for fine-tuning
        for layer in model.layers[:-4]:
            layer.trainable = False
        for layer in model.layers[-4:]:
            layer.trainable = True

        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Lower learning rate for transfer learning
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=4,
            learning_rate=0.0001
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)