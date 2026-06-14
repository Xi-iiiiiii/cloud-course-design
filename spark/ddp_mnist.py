"""
=============================================================================
附加题3-C1：分布式 AI 训练 — PyTorch DDP MNIST CNN
单机 1 进程 vs 2 进程（模拟分布式）训练时间对比 + AllReduce 分析
=============================================================================
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# ============================================================================
# CNN 模型定义（轻量级，CPU 可跑）
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 训练函数（共享）
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device, rank=0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = model(data).argmax(1)
        correct += pred.eq(target).sum().item()
        total += len(target)
    acc = 100.0 * correct / total
    return total_loss / len(loader), acc


# ============================================================================
# DDP Worker（多进程入口）
# ============================================================================
def ddp_worker(rank, world_size, results_queue):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组（Gloo 后端，Windows/CPU 均可用）
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # DistributedSampler 将数据按 rank 分片，每个 Worker 只处理自己那份
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_set, batch_size=64, sampler=sampler)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    t0 = time.time()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        loss, acc = train_epoch(model, loader, optimizer, criterion, "cpu", rank)
        if rank == 0:
            print(f"  [分布式] Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.1f}%")

    elapsed = time.time() - t0
    results_queue.put(elapsed)
    torch.distributed.destroy_process_group()


# ============================================================================
# 单机训练
# ============================================================================
def train_single():
    print("\n[1/2] 单机训练 (1 进程, 全量数据)...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    t0 = time.time()
    for epoch in range(2):
        loss, acc = train_epoch(model, loader, optimizer, criterion, "cpu")
        print(f"  [单机]   Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.1f}%")
    elapsed = time.time() - t0
    print(f"  单机训练耗时: {elapsed:.1f}s")
    return elapsed


# ============================================================================
# 分布式训练（spawn 2 进程）
# ============================================================================
def train_distributed():
    print("\n[2/2] 分布式训练 (2 进程, 数据各分一半)...")
    world_size = 2
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=ddp_worker, args=(rank, world_size, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    times = [queue.get() for _ in range(world_size)]
    max_time = max(times)
    print(f"  分布式训练耗时(取最慢Worker): {max_time:.1f}s")
    return max_time


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("附加题3-C1: 分布式 AI 训练性能对比")
    print("=" * 60)
    print(f"设备: CPU ({os.cpu_count()} 核)")
    print("模型: SimpleCNN (2层卷积, 约 55K 参数)")
    print("数据集: MNIST (60K 训练样本)")
    print("训练: 2 个 epoch, batch_size=64, Adam 优化器")
    print("=" * 60)

    t_single = train_single()
    t_dist = train_distributed()

    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    print(f"  单机 (1 进程, 全量数据):  {t_single:.1f}s")
    print(f"  分布式 (2 进程, 数据分片): {t_dist:.1f}s")
    speedup = t_single / t_dist if t_dist > 0 else 0
    print(f"  加速比: {speedup:.2f}x")
    print(f"  注: CPU 训练下 DDP 的进程间通信(Gloo AllReduce)开销较大,")
    print(f"      小模型(55K 参数)的梯度同步耗时可能超过计算收益。")
    print(f"      在 GPU + NCCL 环境下加速比更接近线性。")
    print("=" * 60)
