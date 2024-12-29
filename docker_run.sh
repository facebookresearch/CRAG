#!/bin/bash

# This script builds a Docker image from the current directory
# and runs a container from this image, executing local_evaluation.py
# with the current directory mounted at /submission inside the container.

# Step 1: Define the name of the Docker image.
LAST_COMMIT_HASH=$(git rev-parse --short HEAD)
IMAGE_NAME="aicrowd/meta-kddcup24-crag-submission:${LAST_COMMIT_HASH}"


# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "Please set the OPENAI_API_KEY environment variable and try again."
    exit 1
fi
# Check if OPENAI_API_KEY is set
if [ -z "$EVALUATION_MODEL_NAME" ]; then
    echo "Warning: EVALUATION_MODEL_NAME is not set."
    echo "Using the default model as: gpt-4-0125-preview"
    export EVALUATION_MODEL_NAME="gpt-4-0125-preview"
fi


# Step 2: Build the Docker image.
# The '.' at the end specifies that the Docker context is the current directory.
# This means Docker will look for a Dockerfile in the current directory to build the image.
START_TIME=$(date +%s)
DOCKER_BUILDKIT=1 docker build -t $IMAGE_NAME .
BUILD_STATUS=$?
if [ $BUILD_STATUS -ne 0 ]; then
    echo "Docker build failed. Exiting..."
    exit $BUILD_STATUS
fi
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))
echo "Total build time: $BUILD_TIME seconds"

# Step 3: Run the Docker container.
# -v "$(pwd)":/submission mounts the current directory ($(pwd) outputs the current directory path)
# to /submission inside the container. This way, the container can access the contents
# of the current directory as if they were located at /submission inside the container.
# 'python /submission/local_evaluation.py' is the command executed inside the container.
# the -w sets the workind directory to /submission.
# It then local_evaluation.py using software runtime set up in the Dockerfile.
docker run \
    --gpus all \
    -v "$(pwd)":/submission \
    -w /submission \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    --ipc=host \
    $IMAGE_NAME python local_evaluation.py

# Note: We assume you have nvidia-container-toolkit installed and configured
# to use the --gpus all flag. If you are not using GPUs, you can remove this flag.


# Note 1: Please refer to the Dockerfile to understand how the software runtime is set up.
# The Dockerfile should include all necessary commands to install Python, the necessary
# dependencies, and any other software required to run local_evaluation.py.

# Note 2: Note the .dockerignore file in the root of this directory.
# In the .dockerignore file, specify any files or directories that should not be included
# in the Docker context. This typically includes large files, models, or datasets that
# are not necessary for building the Docker image. Excluding these can significantly
# speed up the build process by reducing the size of the build context sent to the Docker daemon.

# Ensure your Dockerfile and .dockerignore are properly set up before running this script.
