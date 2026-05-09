docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./models:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
