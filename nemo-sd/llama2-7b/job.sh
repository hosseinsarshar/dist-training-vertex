#!/bin/bash

# to view GPU activities on the web terminal
pip install nvitop

export NCCL_LIB_DIR="/usr/local/nvidia/lib64"
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

# Todo: remove this
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_LOGS="all"
export TORCHDYNAMO_VERBOSE=0

## To turn on debugging
# export TORCH_CPP_LOG_LEVEL=INFO # this is to turn on the verbose torch logs
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

export NODE_RANK=$RANK
export GPUS_PER_NODE=8
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export MASTER_PORT=2222
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*2))

echo "Downloading GPT vocabulary files"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json &&\
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p /tmp/logs/
mkdir -p /tmp/exp/
mkdir -p /tmp/nemo-experiments/results
mkdir -p /tmp/index_mapping_dir

export NODE_RANK=$RANK         
export GPUS_PER_NODE=8
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export MASTER_PORT=2222
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*2))

echo "sleep for 60 seconds"
sleep 10

echo RANK:$RANK
echo NODE_RANK:$NODE_RANK
echo GPUS_PER_NODE:$GPUS_PER_NODE
echo WORLD_SIZE:$WORLD_SIZE
echo MASTER_PORT:$MASTER_PORT
echo NNODES:$NNODES

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"
OMP_NUM_THREADS=12 RANK=$RANK HYDRA_FULL_ERROR=1 \
torchrun  --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --rdzv-backend=static \
    --node_rank=$RANK \
    --rdzv_id $CLOUD_ML_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path="/workspace/a3-bandwidth-test/a3-mega/vertex/nemo/nemo-configs" \
    --config-name="llama2-7b.yaml" \
    +trainer.num_nodes="$NNODES" \
    +exp_manager.explicit_log_dir="/tmp/nemo-experiments/results" \
    +exp_manager.exp_dir="/tmp/exp" \
    +model.data.data_prefix="[]"
