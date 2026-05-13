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

# Build args array
ARGS=("$@")

# Get image digest (optional, you can skip this and just use :latest)
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $DOCKER_USERNAME/$IMAGE_NAME | cut -d'@' -f2)

echo "Using Docker image digest: $IMAGE_DIGEST"

docker run --rm \
  --gpus '"device=0"' \
  -e WANDB_ENTITY=damowerko-academic \
  -e WANDB_USERNAME=shreyas-muthusamy \
  -e WANDB_PROJECT=motion-planning \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v /nfs/general/motion_planning_data:/home/default/motion-planning/data \
  --memory=60g \
  --cpus=16 \
  docker.io/$DOCKER_USERNAME/$IMAGE_NAME@$IMAGE_DIGEST \
  python -u "${ARGS[@]}"
