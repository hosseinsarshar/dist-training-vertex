from mpi4py import MPI
import subprocess
import argparse

def gcloud_storage_copy(src, dest):
    """
    Function to execute the gcloud storage cp command.
    """
    try:
        # Execute the gcloud command to copy files from src to dest
        subprocess.run(["gcloud", "storage", "cp", src, dest], check=True)
        print(f"Successfully copied from {src} to {dest}")
    except subprocess.CalledProcessError as e:
        print(f"Error copying files: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run gcloud storage copy operation with MPI.')
    parser.add_argument('--GCS_PATH_TO_CKPT', type=str, required=True, help='GCS path to checkpoint')
    parser.add_argument('--CONVERTED_MODEL_PATH', type=str, required=True, help='Converted model destination path')

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Rank 0 performs the gcloud storage copy operation before the barrier
        src_location = args.GCS_PATH_TO_CKPT
        dest_location = args.CONVERTED_MODEL_PATH
        
        print(f"Rank 0 is copying from {src_location} to {dest_location}")
        
        # Run the gcloud copy operation (this can take time)
        gcloud_storage_copy(src_location, dest_location)

    # Ensure all processes are synchronized after the copy
    comm.Barrier()

    if rank == 0:
        # Prepare a message to broadcast after the operation is complete
        message = "Copy Complete"
    else:
        message = None

    # Broadcast the message from rank 0 to all other ranks
    message = comm.bcast(message, root=0)

    # All ranks continue with their jobs after receiving the message
    print(f"Rank {rank} received the message: {message}")

    # Continue with the specific job for this rank
    # Your distributed job code here...

if __name__ == "__main__":
    main()
