#!/usr/bin/env bash
# k3s + GPU + KServe (RawDeployment) inside WSL2 Ubuntu — Week 3 Day 1-2.
# Run step by step, not blind. Prereqs: systemd enabled in /etc/wsl.conf, `nvidia-smi` works in WSL.
set -euo pipefail

echo "== 1. nvidia-container-toolkit (containerd needs it) =="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

echo "== 2. k3s (auto-detects nvidia runtime if toolkit precedes it) =="
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube && sudo k3s kubectl config view --raw > ~/.kube/config && chmod 600 ~/.kube/config

echo "== 3. RuntimeClass + NVIDIA device plugin =="
kubectl apply -f - <<'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF
# Device plugin must itself run with the nvidia runtime class:
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
kubectl -n kube-system patch ds nvidia-device-plugin-daemonset \
  --type merge -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'

echo "== 4. GPU smoke test =="
kubectl run cuda-smoke --rm -it --restart=Never \
  --image=nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 \
  --overrides='{"spec":{"runtimeClassName":"nvidia","containers":[{"name":"cuda-smoke","image":"nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}' \
  || echo "SMOKE FAILED — see PLAN.md risk register (fallback: k3d cuda image / minikube --gpus)"

echo "== 5. cert-manager + KServe, RawDeployment mode (no Knative/Istio) =="
KSERVE_VERSION=v0.15.0   # pin what you test
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl wait --for=condition=Available -n cert-manager deploy --all --timeout=300s
helm install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version ${KSERVE_VERSION}
helm install kserve oci://ghcr.io/kserve/charts/kserve --version ${KSERVE_VERSION} \
  --set kserve.controller.deploymentMode=RawDeployment

echo "== 6. ops-overhead ledger inputs =="
free -g
kubectl get pods -A
