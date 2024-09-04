import torch
import torch.distributed as dist
from datetime import timedelta
import os
import time

# Set up a process group with a custom timeout
dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))

# Perform some operations
rank = int(os.getenv('RANK', -1))  # Default to -1 if the RANK variable is not set
local_rank = int(os.environ["LOCAL_RANK"])
print(f"{rank=} - {local_rank=}")

if rank == 0:
    print("Rank 0 is sleeping")
    time.sleep(50)
else:
    print(f"Rank {rank} is waiting")

# Barrier with a timeout in place
dist.barrier()

print("Ranks 0 is sleeping")

# Clean up
dist.destroy_process_group()
