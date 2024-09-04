from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Rank 0 prepares data to broadcast
if rank == 0:
    data = "Hello from Rank 0"
    print(f"Rank 0 broadcasting data: {data}")
else:
    data = None  # All other ranks start with empty data

# Broadcast the data from rank 0 to all ranks
data = comm.bcast(data, root=0)

# Now all ranks should have the same data
print(f"Rank {rank} received data: {data}")
