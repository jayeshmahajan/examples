#!/usr/bin/env python3
"""
PyTorch Distributed Data Parallel (DDP) Training Script (improved)

Fixes and improvements:
- Selects backend based on CUDA availability (nccl/gloo)
- Proper local_rank handling (supports RANK/WORLD_SIZE/LOCAL_RANK env variables from torchrun)
- Avoids using global rank as a device id
- Guards against zero GPU counts and sets device only once
- Ensures checkpoint/final-save are executed only on rank 0
- Moves state_dicts to CPU before saving
- Adds try/finally to ensure cleanup
- Uses pin_memory when CUDA is available
"""

import argparse
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def setup(rank, world_size, backend, master_addr, master_port, local_rank=None):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # If using CUDA, set the correct local device
    if backend == "nccl":
        if local_rank is None:
            raise RuntimeError("local_rank must be provided for NCCL backend")
        torch.cuda.set_device(local_rank)


def cleanup():
    """Clean up the process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def _move_state_to_cpu(state: dict):
    """Recursively move tensors in a state dict to CPU."""
    cpu_state = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.cpu()
        elif isinstance(v, dict):
            cpu_state[k] = _move_state_to_cpu(v)
        else:
            cpu_state[k] = v
    return cpu_state


def train(rank, world_size, local_rank, args):
    """Main training function"""
    print(f"[Rank {rank}] Starting DDP training (world_size={world_size}, local_rank={local_rank})")
    # Choose backend based on CUDA availability
    cuda_available = torch.cuda.is_available()
    backend = "nccl" if cuda_available else "gloo"

    setup(rank, world_size, backend, args.master_addr, args.master_port, local_rank=local_rank if cuda_available else None)

    device = torch.device("cuda", local_rank) if cuda_available else torch.device("cpu")
    print(f"[Rank {rank}] Using device: {device} (backend={backend})")

    try:
        # Create model and move it to device
        model = SimpleCNN().to(device)
        if cuda_available:
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            ddp_model = DDP(model)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

        # Prepare data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Prevent race conditions when downloading
        if rank == 0:
            datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        dist.barrier()  # Wait for rank 0 to finish download
        dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=cuda_available,
            persistent_workers=(args.num_workers > 0)
        )

        # TensorBoard writer (only on rank 0)
        writer = None
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

        # Training loop
        ddp_model.train()
        global_step = 0
        for epoch in range(args.num_epochs):
            sampler.set_epoch(epoch)  # Important for shuffling
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % args.log_interval == 0 and rank == 0:
                    print(f'[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                    if writer:
                        writer.add_scalar('Loss/Train_batch', loss.item(), global_step)
                global_step += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            if rank == 0:
                if writer:
                    writer.add_scalar('Loss/Train_epoch', avg_loss, epoch)
                print(f'[Rank {rank}] Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

                # Save checkpoint only from rank 0
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
                # Move model and optimizer states to CPU before saving
                model_state_cpu = _move_state_to_cpu(ddp_model.module.state_dict())
                optimizer_state_cpu = _move_state_to_cpu(optimizer.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_cpu,
                    'optimizer_state_dict': optimizer_state_cpu,
                    'loss': avg_loss,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f'[Rank {rank}] Checkpoint saved to {checkpoint_path}')

        # Final save (only rank 0)
        if rank == 0:
            final_model_path = os.path.join(args.output_dir, 'final_model.pt')
            model_state_cpu = _move_state_to_cpu(ddp_model.module.state_dict())
            torch.save(model_state_cpu, final_model_path)
            print(f'[Rank {rank}] Final model saved to {final_model_path}')

            if writer:
                writer.close()

    finally:
        # Ensure cleanup even if something fails
        cleanup()
        print(f"[Rank {rank}] Cleanup done.")


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training (improved)')
    parser.add_argument('--data-dir', type=str, default='/data', help='Directory for training data')
    parser.add_argument('--output-dir', type=str, default='/output', help='Directory for outputs and checkpoints')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--world-size', type=int, default=None, help='Total number of processes')
    parser.add_argument('--rank', type=int, default=None, help='Global rank of this process')
    parser.add_argument('--local-rank', type=int, default=None, help='Local CUDA device index for this process')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1', help='Address of the master node')
    parser.add_argument('--master-port', type=int, default=29500, help='Port for distributed communication')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--log-interval', type=int, default=100, help='Batches between log messages')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Allow launcher env variables (torchrun) to override args
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    env_local_rank = os.environ.get("LOCAL_RANK")

    if env_rank is not None:
        rank = int(env_rank)
    elif args.rank is not None:
        rank = int(args.rank)
    else:
        print("Error: rank not provided via args or RANK env var.", file=sys.stderr)
        sys.exit(1)

    if env_world_size is not None:
        world_size = int(env_world_size)
    elif args.world_size is not None:
        world_size = int(args.world_size)
    else:
        print("Error: world_size not provided via args or WORLD_SIZE env var.", file=sys.stderr)
        sys.exit(1)

    if env_local_rank is not None:
        local_rank = int(env_local_rank)
    else:
        # Fallback to provided local-rank arg or compute mod if GPUs exist
        if args.local_rank is not None:
            local_rank = int(args.local_rank)
        else:
            local_rank = 0

    # If CUDA is available, ensure device_count > 0
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("CUDA is reported available but no devices found.", file=sys.stderr)
            sys.exit(1)
        if local_rank >= device_count:
            print(f"local_rank {local_rank} is >= available device count {device_count}", file=sys.stderr)
            sys.exit(1)

    # Ensure output dir exists if rank 0 will write
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    train(rank, world_size, local_rank, args)


if __name__ == '__main__':
    main()