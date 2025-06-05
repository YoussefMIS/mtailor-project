from fastapi import FastAPI, UploadFile, File
from model import OnnxModel, ImagePreprocessor
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Initialize ONNX model and preprocessor
model_path = "mtailor.onnx"
image_size = (224, 224)
onnx_model = OnnxModel(model_path)
preprocessor = ImagePreprocessor(image_size)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocessor.preprocess(image)
    input_data = np.expand_dims(input_tensor.numpy(), axis=0)  # Add batch dimension

    # Perform inference using the ONNX model
    prediction = onnx_model.predict(input_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return {"prediction": predicted_class}
