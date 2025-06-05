# Mtailor Interview Project

This repository contains a machine learning model deployment project. The project includes preprocessing, ONNX model conversion, deployment using FastAPI, and testing on the Cerebrium platform.

## Prerequisites

1. **Python**: Ensure Python 3.13 is installed.
2. **Docker**: Install Docker to build and run the containerized application.
3. **Pip**: Ensure `pip` is installed and updated.
4. **Cerebrium Account**: Set up an account on Cerebrium for deployment.

## Repository Structure

- `convert_to_onnx.py`: Script to convert a PyTorch model to ONNX format.
- `main.py`: FastAPI application for serving the ONNX model.
- `model.py`: Contains the ONNX model wrapper and image preprocessing logic.
- `test.py`: Unit tests for local model and preprocessing.
- `test_server.py`: Tests for the deployed model on Cerebrium.
- `requirements.txt`: Python dependencies.
- `Dockerfile.txt`: Dockerfile for containerizing the application.
- `cerebrium.toml`: Configuration file for Cerebrium deployment.

## Steps to Run the Project

### 1. Install Dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Convert PyTorch Model to ONNX

Use the `convert_to_onnx.py` script to convert the PyTorch model to ONNX format:

```bash
python convert_to_onnx.py
```

This will generate the `mtailor.onnx` file in the project directory.

### 3. Run the FastAPI Application Locally

Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8192
```

The server will be accessible at `http://127.0.0.1:8192`.

### 4. Test the Application Locally

Run the unit tests to verify the application:

```bash
python -m unittest test.py
```

### 5. Deploy to Cerebrium

1. Ensure the `cerebrium.toml` file is correctly configured.
2. Use the Cerebrium CLI to deploy the application:

```bash
cerebrium deploy
```

### 6. Test the Deployed Model

Use the `test_server.py` script to test the deployed model:

```bash
python test_server.py --image <path_to_image>
```

Replace `<path_to_image>` with the path to the image you want to test.

To run preset custom tests:

```bash
python test_server.py --run-custom-tests
```

### 7. Monitor the Deployed Model

The `test_server.py` script also includes monitoring functionality. Use the following command to check the health and response time of the deployed model:

```bash
python test_server.py --monitor
```

## Notes

- Ensure the `mtailor.onnx` file is included in the deployment.
- Update the `CEREBRIUM_MODEL_ENDPOINT` in `test_server.py` with the actual endpoint URL after deployment.
- Use the `Dockerfile.txt` to build a Docker image for the application if needed.

## Contact

For any issues or questions, please contact me