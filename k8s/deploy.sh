#!/bin/bash
# =============================================================================
# 云计算课程设计 - 应用部署脚本
# 一键部署第一部分 Flask+Redis 应用到华为云 CCE 集群
#
# 使用方法:
#   1. 确保已配置 kubectl (CCE 控制台下载 KubeConfig)
#   2. 修改 k8s/ 目录下 YAML 中 cloud-course-2023112435 等占位符
#   3. 运行: bash k8s/deploy.sh
# =============================================================================

set -e

echo "=============================================="
echo "  云计算课程设计 - 应用部署"
echo "=============================================="

# ── 1. 创建 Secret（Redis 密码）────────────────────
echo "[1/6] 创建 Secret..."
kubectl apply -f k8s/secret.yaml

# ── 2. 创建 ConfigMap ──────────────────────────────
echo "[2/6] 创建 ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# ── 3. 创建 PVC（持久化存储）───────────────────────
echo "[3/6] 创建 PVC..."
kubectl apply -f k8s/pvc.yaml

# ── 4. 部署应用 ────────────────────────────────────
echo "[4/6] 部署应用 (Deployment + Service)..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# ── 5. 配置 HPA（弹性伸缩）─────────────────────────
echo "[5/6] 创建 HPA..."
kubectl apply -f k8s/hpa.yaml

# ── 6. 等待部署就绪 ────────────────────────────────
echo "[6/6] 等待 Pod 就绪..."
kubectl wait --for=condition=ready pod -l app=backend --timeout=120s
kubectl wait --for=condition=ready pod -l app=frontend --timeout=120s

echo ""
echo "=============================================="
echo "  部署完成！验证命令："
echo "=============================================="
echo "  kubectl get pods -o wide"
echo "  kubectl get svc backend-svc  # 获取 ELB 公网 IP"
echo "  curl http://<ELB_IP>/api/ping"
echo "=============================================="
