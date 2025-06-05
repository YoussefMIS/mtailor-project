import requests
import unittest
import argparse
import time

# Replace with your deployed model's endpoint
CEREBRIUM_MODEL_ENDPOINT = "https://your-cerebrium-model-endpoint.com/predict"


def call_deployed_model(image_path):
    """
    Calls the deployed model on Cerebrium with the given image path.
    Returns the class ID predicted by the model.
    """
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(CEREBRIUM_MODEL_ENDPOINT, files=files)
        response.raise_for_status()
        return response.json().get('prediction')


def monitor_deployed_model():
    """
    Monitor the deployed model's health and response time.
    """
    start_time = time.time()
    response = requests.get(f"{CEREBRIUM_MODEL_ENDPOINT}/monitor")
    response_time = time.time() - start_time
    response.raise_for_status()
    return response.json(), response_time


class TestDeployedModel(unittest.TestCase):
    def test_valid_image(self):
        """Test the deployed model with a valid image."""
        image_path = "n01667114_mud_turtle.JPEG"  # Replace with a valid test image path
        class_id = call_deployed_model(image_path)
        self.assertIsNotNone(class_id, "The class ID should not be None.")
        self.assertIsInstance(class_id, int, "The class ID should be an integer.")

    def test_invalid_image(self):
        """Test the deployed model with an invalid image."""
        image_path = "invalid_image.txt"  # Replace with an invalid test file path
        with self.assertRaises(requests.exceptions.RequestException):
            call_deployed_model(image_path)

    def test_missing_image(self):
        """Test the deployed model with a missing image."""
        image_path = "non_existent_image.jpg"  # Replace with a non-existent file path
        with self.assertRaises(FileNotFoundError):
            call_deployed_model(image_path)

    def test_monitoring_endpoint(self):
        """Test the Cerebrium platform's monitoring endpoint."""
        monitoring_data, response_time = monitor_deployed_model()
        self.assertEqual(monitoring_data.get("status"), "healthy", "Model should be healthy.")
        self.assertLess(response_time, 2, "Response time should be less than 2 seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the deployed model.")
    parser.add_argument("--image", type=str, help="Path to the image to test.")
    parser.add_argument("--run-custom-tests", action="store_true", help="Run preset custom tests.")
    args = parser.parse_args()

    if args.image:
        try:
            class_id = call_deployed_model(args.image)
            print(f"Predicted class ID: {class_id}")
        except Exception as e:
            print(f"Error: {e}")
    elif args.run_custom_tests:
        unittest.main(argv=[''], exit=False)
    else:
        print("Please provide an image path or use the --run-custom-tests flag.")