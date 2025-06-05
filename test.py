import unittest
from model import ImagePreprocessor, OnnxModel  # Explicit imports instead of star imports
from PIL import Image
import numpy as np  # Explicit import for numpy
import torch

image_path = "./n01667114_mud_turtle.JPEG"  # Example image path
img = Image.open(image_path)

class TestModelDeployment(unittest.TestCase):
    def setUp(self):
        # Set up common test variables
        self.image_size = (224, 224)
        self.model_path = "mtailor.onnx"
        self.preprocessor = ImagePreprocessor(self.image_size)
        self.onnx_model = OnnxModel(self.model_path)

    def test_image_preprocessor_initialization(self):
        # Test if the ImagePreprocessor initializes correctly
        self.assertEqual(self.preprocessor.image_size, self.image_size)

    def test_onnx_model_initialization(self):
        # Test if the OnnxModel initializes correctly
        self.assertEqual(self.onnx_model.model_path, self.model_path)

    def test_preprocessor_process_image(self):
        # Test the image preprocessing functionality
        processed_image = self.preprocessor.preprocess(img)
        self.assertIsNotNone(processed_image)
        # Test if the image is resized correctly
        self.assertEqual(processed_image.shape, (3, 224, 224))

    def test_onnx_model_inference(self):
        # Test the ONNX model inference
        processed_input = self.preprocessor.preprocess(img)
        processed_input = np.expand_dims(processed_input.numpy(), axis=0)
        prediction = self.onnx_model.predict(processed_input)
        self.assertIsNotNone(prediction)

        expected_class = 35
        predicted_class = np.argmax(prediction)
        self.assertEqual(predicted_class, expected_class)

if __name__ == "__main__":
    unittest.main()