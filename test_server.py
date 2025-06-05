import requests
import unittest
import argparse
import time
import logging
import base64

# Update the endpoint to match the FastAPI application in main.py
CEREBRIUM_MODEL_ENDPOINT = "https://api.cortex.cerebrium.ai/v4/p-68e74239/my-first-project/predict"

# Add an authentication token for the external endpoint
AUTH_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTY4ZTc0MjM5IiwiaWF0IjoxNzQ5MTIzMTAxLCJleHAiOjIwNjQ2OTkxMDF9.B1rr50ABFF4QsBqgDYLvikTtFsf_J-iwv5LunFN7kvba7t2CruYcBMYbvO5-rvQm7QFar1VSgEqIICFlmITE_sEomGXoMbJRbcKsBhllcaXI4GxukCfv1hswTs_54XOQj4Rrhpd7O0IYX7X6t9hgcgCPYaAGX1AXnl4zHJzFnxCox8XklUDla7HpigihvDqCr2yIA5cIjwy3fm_AkD4o64TO9lBmxg-IuQtdhVW9Vr3vJj3urX4zRZxlp2Dqx1SMIPiNbZ3b5SHv92DpcJpcCcdXLo5FbFEM0z6YxPnfW2OdTd8PjUORsaCSC8N0-Cv_adLZuzKpxxxb3D1V6zoZ5g"  # Replace with the actual token

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def call_deployed_model(image_path):
    """
    Calls the deployed model on the local FastAPI server with the given image path.
    Returns the class ID predicted by the model.
    """
    try:
        with open(image_path, 'rb') as image_file:
            # Encode the image as a base64 string
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            headers = {
                'Authorization': f'Bearer {AUTH_TOKEN}',
                'Content-Type': 'text/plain'  # Set content type to plain text
            }
            logging.debug(f"Sending request to {CEREBRIUM_MODEL_ENDPOINT} with headers: {headers} and base64 image data")
            response = requests.post(CEREBRIUM_MODEL_ENDPOINT, data=image_base64, headers=headers)
            logging.debug(f"Response status code: {response.status_code}")
            logging.debug(f"Response content: {response.text}")
            response.raise_for_status()
            return response.json().get('prediction')
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        raise


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
        image_path = "model.py"  # Replace with an invalid test file path
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