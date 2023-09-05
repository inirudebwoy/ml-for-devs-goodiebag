import io

from fastapi import FastAPI, UploadFile
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


@app.post("/what")
async def what_is_it(img: UploadFile = None):
    contents = await img.read()
    image = Image.open(io.BytesIO(contents))

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return {"Predicted class": model.config.id2label[predicted_class_idx]}
