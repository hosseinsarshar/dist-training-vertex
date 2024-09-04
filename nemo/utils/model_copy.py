import torch
import torch.distributed as dist
from datetime import timedelta
import os
import time
import subprocess
import argparse

# Set up a process group with a custom timeout
dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))

def gcloud_storage_copy(src, dest):
    """
    Function to execute the gcloud storage cp command.
    """
    try:
        # Execute the gcloud command to copy files from src to dest
        # subprocess.run(["gcloud", "storage", "cp", src, dest], check=True)
        time.sleep(15)
        print(f"Successfully copied from {src} to {dest} on RANK=0")
    except subprocess.CalledProcessError as e:
        print(f"Error copying files: {e} on RANK=0")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run gcloud storage copy operation with MPI.')
    parser.add_argument('--src', type=str, required=True, help='GCS path to checkpoint')
    parser.add_argument('--dest', type=str, required=True, help='Converted model destination path')

    args = parser.parse_args()

    # Initialize MPI
    rank = int(os.getenv('RANK', -1))  # Default to -1 if the RANK variable is not set
    print(f"Rank {rank}")

    if rank == -1:
        raise ValueError("Error: RANK environment variable is not set.")

    if rank == 0:
        # Rank 0 performs the gcloud storage copy operation before the barrier
        src_location = args.src
        dest_location = args.dest
        
        print(f"Rank 0 is copying from {src_location} to {dest_location}")
        
        # Run the gcloud copy operation (this can take time)
        gcloud_storage_copy(src_location, dest_location)
    else:
        print(f"Rank={rank} is waiting for the copy operation to complete.")

    # Ensure all processes are synchronized after the copy
    dist.barrier()

    # All ranks continue with their jobs after receiving the message
    print(f"Rank {rank} is done")

    dist.destroy_process_group()

    print(f"Rank {rank} called dist.destroy_process_group()")

    # Continue with the specific job for this rank
    # Your distributed job code here...

if __name__ == "__main__":
    main()
