import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

class OnnxModel:
    def __init__(self, model_path):
        """
        Initialize the ONNX model.
        :param model_path: Path to the ONNX model file.
        """
        self.session = ort.InferenceSession(model_path)
        self.model_path = model_path
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        """
        Perform prediction using the ONNX model.
        :param input_data: Pre-processed input data.
        :return: Model prediction.
        """
        return self.session.run([self.output_name], {self.input_name: input_data})[0]


class ImagePreprocessor:
    def __init__(self, image_size):
        """
        Initialize the image preprocessor.
        :param image_size: Tuple (width, height) for resizing the image.
        """
        self.image_size = image_size

    def preprocess(self, img):
        """
        Pre-process the image for model input.
        :param img: the image to be processed.
        :return: Pre-processed image as a numpy array.
        """
        resize = transforms.Resize((224, 224))   #must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #print the format of the image
        print(img.format)
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img
    


if __name__ == "__main__":
    # Example usage
    model_path = "mtailor.onnx"
    image_path = "./n01667114_mud_turtle.JPEG"
    image_size = (224, 224)
    preprocessor = ImagePreprocessor(image_size)
    onnx_model = OnnxModel(model_path)
    img = Image.open(image_path).convert('RGB')
    input_data = preprocessor.preprocess(img)
    input_data = np.expand_dims(input_data.numpy(), axis=0)  # Add batch dimension
    prediction = onnx_model.predict(input_data)
    #getting the class with highest probability
    print("Prediction:", np.argmax(prediction, axis=1)[0])