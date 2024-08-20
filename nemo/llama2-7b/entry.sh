#!/bin/bash

echo "Running on a host: $(hostname)"

: "${GPUS_PER_NODE:?Must set GPUS_PER_NODE}"
: "${NCCL_LIB_DIR}:?Must set NCCL_LIB_DIR}"
: "${JOB_ID}:?Must set JOB_ID}"
: "${OUTPUT_GCS_PATH_PREFIX}:?Must set OUTPUT_GCS_PATH_PREFIX}"
: "${GCS_DATA_SOURCE}:?Must set GCS_DATA_SOURCE}"
: "${DATA_PREFIX}:?Must set DATA_PREFIX}"
: "${CONFIG_NAME}:?Must set CONFIG_NAME}"
: "${TORCH_DISTRIBUTED_TARGET}:?Must set TORCH_DISTRIBUTED_TARGET}"


########

export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_FASTRAK_CTRL_DEV=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=${NCCL_LIB_DIR}/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=${NCCL_LIB_DIR}/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_PXN_LEVEL=0
export NCCL_DEBUG=VERSION
export NNODES="${WORLD_SIZE}"
#export NCCL_FASTRAK_NUM_FLOWS_PER_GROUP=1        
#export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
#export GLOO_SOCKET_IFNAME="eth0"

ldconfig ${NCCL_LIB_DIR}

master_addr="${MASTER_ADDR}"
master_port="$MASTER_PORT"
node_rank="$RANK"
nnodes="$WORLD_SIZE"

echo "Running torchrun on Rank: ${RANK} using the following settings:"
echo "NODE_RANK: ${node_rank}"
echo "MASTER_ADDR: ${master_addr}"
echo "MASTER_PORT: ${master_port}"
echo "NNODES: ${nnodes}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--node_rank ${node_rank} \
--rdzv_backend=static \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py

exit 0

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--node_rank ${node_rank} \
--master_addr ${master_addr} \
--master_port ${master_port} \
hello_world.py

exit 0

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
--rdzv-conf=is_host=$(if ((RANK)); then echo 0; else echo 1; fi) \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py




exit 0
###############

sleep_seconds=infinity
#sleep_seconds=30
echo "Sleeping for ${sleep_seconds}"
sleep ${sleep_seconds}

echo "Downloading training datasets from ${GCS_DATA_SOURCE} to /dataset"
mkdir -p /dataset
gcloud storage rsync \
--recursive \
$GCS_DATA_SOURCE /dataset 

echo "Downloading GPT vocabulary files"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json &&\
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

export NCCL_FASTRAK_IFNAME=eth1,eth2,eth3,eth4,eth5,eth6,eth7,eth8
export NCCL_FASTRAK_CTRL_DEV=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=${NCCL_LIB_DIR}/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=${NCCL_LIB_DIR}/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_PXN_LEVEL=0
export NCCL_DEBUG=VERSION
export NNODES="${WORLD_SIZE}"
#export NCCL_FASTRAK_NUM_FLOWS_PER_GROUP=1        
#export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
#export GLOO_SOCKET_IFNAME="eth0"

ldconfig ${NCCL_LIB_DIR}

master_addr="${MASTER_ADDR}"
master_port="$MASTER_PORT"
node_rank="$RANK"
nnodes="$WORLD_SIZE"

echo "Running torchrun on Rank: ${RANK} using the following settings:"
echo "NODE_RANK: ${node_rank}"
echo "MASTER_ADDR: ${master_addr}"
echo "MASTER_PORT: ${master_port}"
echo "NNODES: ${nnodes}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"

data_prefix="[1.0,/dataset/${DATA_PREFIX}]"
output_gcs_prefix="/gcs/${OUTPUT_GCS_PATH_PREFIX:5}"
index_mapping_dir="${output_gcs_prefix}/${JOB_ID}/index_mapping_dir"
mkdir -p ${index_mapping_dir}
dllogger_path="${output_gcs_prefix}/${JOB_ID}/logs/${RANK}/dllogger.json"

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--node_rank ${node_rank} \
--master_addr ${master_addr} \
--master_port ${master_port} \
${TORCH_DISTRIBUTED_TARGET} \
--config-path="/workspace" \
--config-name="${CONFIG_NAME}" \
+trainer.num_nodes="${nnodes}" \
+model.data.data_prefix="${data_prefix}" \
+model.data.index_mapping_dir="${index_mapping_dir}" \
+exp_manager.version="${JOB_ID}" \
+exp_manager.dllogger_logger_kwargs.json_file="${dllogger_path}"  


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--node_rank ${node_rank} \
--master_addr ${master_addr} \
--master_port ${master_port} \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--master_addr ${master_addr} \
--master_port ${master_port} \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--node_rank ${node_rank} \
--rdzv_backend=static \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
--rdzv-conf=is_host=$(if ((RANK)); then echo 0; else echo 1; fi) \
--local_addr=${pod_hostname} \
hello_world.py

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
--rdzv-conf=is_host=$(if ((RANK)); then echo 0; else echo 1; fi) \
hello_world.py

OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="${master_addr}:${master_port}" \
hello_world.py


OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=${GPUS_PER_NODE} \
--nnodes=${nnodes} \
--rdzv_id=101 \
--rdzv_backend=c10d \
--rdzv-endpoint="localhost:${master_port}" \
hello_world.py