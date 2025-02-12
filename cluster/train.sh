#!/bin/bash
set -e
$(dirname "$0")/build.sh
args="$@"
IMAGE_NAME="motion-planning"
DOCKER_USERNAME="damowerko"
kubectl create -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: motion-planning-train-
  namespace: owerko
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
        command: ["bash", "-c", "python -u scripts/train.py $args --no_bar && sleep 1"]
        env:
        - name: WANDB_ENTITY
          value: damowerko-academic
        - name: WANDB_USERNAME
          value: damowerko
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