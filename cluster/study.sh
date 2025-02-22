#!/bin/bash
set -e
completions="$1"
shift
if ! [[ "$completions" =~ ^[0-9]+$ ]]; then
  echo "First argument must be a number"
  exit 1
fi
args="$@"
IMAGE_NAME="motion-planning"
K8S_NAMESPACE=${K8S_NAMESPACE:-$(kubectl config get-contexts | grep '*' | awk '{print $5}')}
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME environment variable is not set"
    exit 1
fi

template=$(cat << EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: motion-planning-optuna-
  namespace: $K8S_NAMESPACE
spec:
  completions: $completions
  parallelism: 8
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: motion-planning-trial
        image: docker.io/$DOCKER_USERNAME/$IMAGE_NAME
        imagePullPolicy: Always
        command: ["bash", "-c", "python -u scripts/study.py $args --no_bar && sleep 1"]
        env:
        - name: OPTUNA_STORAGE
          value: postgresql://optuna:optuna@optuna-db.owerko.svc.cluster.local:5432/optuna
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
            cpu: 8
            memory: 16Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 64
            memory: 128Gi
            nvidia.com/gpu: 1
EOF
)
echo "$template" | kubectl create -f -