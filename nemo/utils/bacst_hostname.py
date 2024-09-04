from mpi4py import MPI
import subprocess

def get_hostname():
    # Run 'uname -n' to get the hostname
    result = subprocess.run(['uname', '-n'], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the hostname of the current node
hostname = get_hostname()

# Rank 0 will gather the hostnames from all ranks
if rank == 0:
    hostnames = [None] * size  # Pre-allocate list to gather all hostnames
else:
    hostnames = None  # Other ranks don't need the list

# Gather all hostnames at rank 0
hostnames = comm.gather(hostname, root=0)

# Rank 0 prints all the hostnames
if rank == 0:
    for i, host in enumerate(hostnames):
        print(f"Rank {i} is running on host: {host}")
