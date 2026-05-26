#!/bin/bash
set -e
IMAGE_NAME="motion-planning"

if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME environment variable is not set"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set"
    exit 1
fi

# Build argument list
printf -v args "\"%s\" " "$@"

# Get the current digest of the Docker image
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $DOCKER_USERNAME/$IMAGE_NAME | cut -d'@' -f2)
echo "Using Docker image digest: $IMAGE_DIGEST"

# Run the container
docker run --rm \
  --gpus '"device=0"' \
  --cpus="16" \
  --memory="32g" \
  -e WANDB_ENTITY=damowerko-academic \
  -e WANDB_USERNAME=shreyas-muthusamy \
  -e WANDB_PROJECT=motion-planning \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  docker.io/$DOCKER_USERNAME/$IMAGE_NAME@$IMAGE_DIGEST \
  python -u scripts/train.py $@ --simple_progress
