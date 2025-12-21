import jukebox_modules.utils.dist_adapter as dist
import torch


def print_once(msg):
    if (not dist.is_available()) or dist.get_rank() == 0:
        print(msg)


def print_all(msg):
    if not dist.is_available():
        print(msg)
    elif dist.get_rank() % 8 == 0:
        print(f"{dist.get_rank() // 8}: {msg}")


def allreduce(x, op=dist.ReduceOp.SUM):
    """Allreduce for distributed training (stub - not used in inference)."""
    x = torch.tensor(x).float().cuda()
    dist.all_reduce(x, op=op)
    return x.item()
