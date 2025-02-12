IMAGE_NAME="motion-planning"
DOCKER_USERNAME="damowerko"
template=$(cat << EOF
apiVersion: v1
kind: Pod
metadata:
  name: motion-planning-interactive
  namespace: owerko
spec:
  containers:
  - name: motion-planning-shell
    image: docker.io/$DOCKER_USERNAME/$IMAGE_NAME
    imagePullPolicy: Always
    command: ["/bin/bash"]
    tty: true
    stdin: true
    env:
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
)
echo "$template" | kubectl create -f -