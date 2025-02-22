#! /bin/bash
set -e
IMAGE_NAME="motion-planning"
K8S_NAMESPACE=${K8S_NAMESPACE:-$(kubectl config get-contexts | grep '*' | awk '{print $5}')}
if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DOCKER_USERNAME environment variable is not set"
    exit 1
fi
template=$(cat << EOF
apiVersion: v1
kind: Pod
metadata:
  name: motion-planning-interactive
  namespace: $K8S_NAMESPACE
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
kubectl wait --for=condition=Ready pod/motion-planning-interactive --namespace=$K8S_NAMESPACE --timeout=300s
kubectl exec -it motion-planning-interactive --namespace=$K8S_NAMESPACE -- /bin/bash