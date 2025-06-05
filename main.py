from fastapi import FastAPI
from pydantic import BaseModel
from model import OnnxModel, ImagePreprocessor
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()

# Initialize ONNX model and preprocessor
model_path = "mtailor.onnx"
image_size = (224, 224)
onnx_model = OnnxModel(model_path)
preprocessor = ImagePreprocessor(image_size)

class ImagePayload(BaseModel):
    payload: str  # Base64-encoded image string

@app.post("/predict")
async def predict(payload: ImagePayload):
    try:
        # Decode the base64-encoded image
        image_bytes = base64.b64decode(payload.payload)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocessor.preprocess(image)
        input_data = np.expand_dims(input_tensor.numpy(), axis=0)  # Add batch dimension

        # Perform inference using the ONNX model
        prediction = onnx_model.predict(input_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}, 400
