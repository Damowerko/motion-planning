#!/bin/bash
set -e
IMAGE_NAME="motion-planning"
DOCKER_USERNAME="shreyasmuthusamy"

# comma separated list of arguments, printf adds an extra comma at the end, so we remove it
printf -v args "\"%s\"," "$@"
args=${args%,}

# build first
$(dirname "$0")/build.sh

# create the job
kubectl create -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: motion-planning-train-
  namespace: muthurak
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: motion-planning-train
        image: docker.io/$DOCKER_USERNAME/$IMAGE_NAME
        imagePullPolicy: Always
        command: ["python", "-u", "scripts/train.py", $args, "--no_bar"]
        env:
        - name: WANDB_ENTITY
          value: damowerko-academic
        - name: WANDB_USERNAME
          value: shreyas-muthusamy
        - name: WANDB_PROJECT
          value: motion-planning
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb
              key: api_key
        resources:
          requests:
            cpu: 16
            memory: 32Gi
          limits:
            nvidia.com/gpu: 1
EOF