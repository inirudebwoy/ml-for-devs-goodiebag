import base64
import io
import json
import logging
import os
from pathlib import Path

from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global processor

    processor = ViTImageProcessor.from_pretrained(
        os.getenv("AZUREML_MODEL_DIR") / Path("vit-base-patch16-224/"), from_flax=True
    )
    model = ViTForImageClassification.from_pretrained(
        os.getenv("AZUREML_MODEL_DIR") / Path("vit-base-patch16-224/"), from_flax=True
    )
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    data = json.loads(raw_data)["data"]
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    inputs = processor(images=image, return_tensors="pt")
    result = model(**inputs)
    logging.info("Request processed")
    logits = result.logits
    predicted_class_idx = logits.argmax(-1).item()
    return {"Predicted class": model.config.id2label[predicted_class_idx]}
