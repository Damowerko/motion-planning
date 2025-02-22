#!/bin/bash
set -e
IMAGE_NAME="motion-planning"
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME environment variable is not set"
    exit 1
fi
export DOCKER_BUILDKIT=1
docker pull docker.io/$DOCKER_USERNAME/$IMAGE_NAME:latest || true
docker build . -t docker.io/$DOCKER_USERNAME/$IMAGE_NAME --build-arg BUILDKIT_INLINE_CACHE=1 
docker push docker.io/$DOCKER_USERNAME/$IMAGE_NAME
