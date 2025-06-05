from fastapi import FastAPI, HTTPException, Request
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

@app.post("/predict")
async def predict(request: Request):
    try:
        # Read the base64-encoded image string directly from the request body
        image_base64 = await request.body()
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocessor.preprocess(image)
        input_data = np.expand_dims(input_tensor.numpy(), axis=0)  # Add batch dimension

        # Perform inference using the ONNX model
        prediction = onnx_model.predict(input_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
