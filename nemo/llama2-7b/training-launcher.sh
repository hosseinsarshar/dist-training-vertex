#!/bin/bash

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
# export NCCL_DEBUG=INFO

python -c "print('Number of nodes participating: 2')"
echo NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS: $NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS
echo MASTER_ADDR: $MASTER_ADDR
echo LOCAL_RANK: $LOCAL_RANK
echo JOB_COMPLETION_INDEX: $JOB_COMPLETION_INDEX


function on_script_completion {
    # Note: This semaphore is used to terminate the TCPx side-car
    touch /semaphore/workload_terminated
}

export CLOUD_ML_JOB_ID=123
export JOB_IDENTIFIER=nemo-vertex-$CLOUD_ML_JOB_ID

# trap on_script_completion EXIT
echo "Pod on $(hostname --fqdn) is running"
echo "Pod is assigned job index of $JOB_COMPLETION_INDEX"
echo "Job ID is $JOB_IDENTIFIER"

echo "Running nvidia-smi"
nvidia-smi

# mkdir -p /tmp
# gcsfuse --client-protocol http2 $GCS_FUSE_BUCKET /tmp 

# mkdir -p /tmp/index_mapping_dir

# export LD_LIBRARY_PATH="/usr/local/nccl-plugin/lib64:/usr/local/cuda-12.3/lib64:/usr/local/nvidia/lib64/:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/nccl-plugin/lib64:/usr/local/nvidia/lib64/:${LD_LIBRARY_PATH}"
echo "Warning: Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH to override the NCCL library"

ldconfig /usr/local/nvidia/lib64/
echo "Added /usr/local/nvidia/lib64/ to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'

echo "Contents of /usr/local/nccl-plugin/lib64:"
ls /usr/local/nccl-plugin/lib64 | sed 's/^/  /'

export SSD_MOUNT_PATH=/tmp/ssd

mkdir -p $SSD_MOUNT_PATH

touch $SSD_MOUNT_PATH/hello-from-$HOSTNAME.txt
echo "Local SSD contents (path $SSD_MOUNT_PATH):"; ls $SSD_MOUNT_PATH | sed 's/^/  /'



echo "Downloading GPT vocabulary files"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json &&\
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

git clone https://github.com/hosseinsarshar/a3-bandwidth-test.git

echo "NeMo configuration file:"                                         
cat a3-bandwidth-test/a3-mega/vertex/nemo/nemo-configs/llama7b-bf16-16gpus.yaml | sed 's/^/| /' 
echo ""
readarray -d "" workload_arguments < <(env | grep -e "^WORKLOAD_" | sed 's/^WORKLOAD_/+/' | tr '\n' '\0') 
echo "Detected the following additional workload arguments:"            
for workload_argument in "${workload_arguments[@]}"; do                 
    echo "  $workload_argument"                                           
done 

sleep 10 # <- Hack to allow some time for service to boot

mount /tmp -o remount,exec 
chmod -R a+rwx /tmp

echo "Checking for presence of nsys:"                                   
which nsys  

echo "Nsight profiling will go to /tmp/hosseins-vertex-test/$JOB_IDENTIFIER/."
mkdir -p /tmp/hosseins-vertex-test/$JOB_IDENTIFIER/

# apt -y update && apt -y install gdb python3.10-dbg

python -c "import os; print(os.listdir('.'))"

mkdir -p /tmp/logs/
mkdir -p /tmp/exp/
mkdir -p /tmp/nemo-experiments/results
mkdir -p /tmp/index_mapping_dir

export NODE_RANK=$RANK         
export GPUS_PER_NODE=8
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export MASTER_PORT=2222
export GLOBAL_BATCH_SIZE=$((WORLD_SIZE*2))
# export MASTER_ADDR=localhost

echo "sleep for 60 seconds"
sleep 60
echo RANK:$RANK
echo NODE_RANK:$NODE_RANK
echo GPUS_PER_NODE:$GPUS_PER_NODE
echo WORLD_SIZE:$WORLD_SIZE
echo MASTER_PORT:$MASTER_PORT
echo NNODES:$NNODES
# echo GLOBAL_BATCH_SIZE:$GLOBAL_BATCH_SIZE
echo rdzv_endpoint=$(if [[ $RANK -gt 0 ]]; then echo $MASTER_ADDR;else echo localhost;fi):$MASTER_PORT
# torchrun --rdzv_backend c10d --rdzv_id $CLOUD_ML_JOB_ID --nnodes 2 --nproc_per_node 8 --rdzv_endpoint=

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"
# OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK HYDRA_FULL_ERROR=1 \

OMP_NUM_THREADS=12 RANK=$RANK HYDRA_FULL_ERROR=1 \
torchrun  --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --rdzv-backend=static \
    --node_rank=$RANK \
    --rdzv_id $CLOUD_ML_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    NemoHossein/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path="/workspace/dist-training-vertex/nemo/llama2-7b/" \
    --config-name="llama7b-bf16-16gpus.yaml" \
    +trainer.num_nodes="$NNODES" \
    +exp_manager.explicit_log_dir="/tmp/nemo-experiments/results" \
    +exp_manager.version="$JOB_IDENTIFIER" \
    +exp_manager.exp_dir="/tmp/exp" \
    +model.data.data_prefix="[]"
    # \
    # ++model.global_batch_size="$GLOBAL_BATCH_SIZE"

    # torchrun  --nproc_per_node=${GPUS_PER_NODE} \
    #     --nnodes=${NNODES} \
    #     --rdzv_backend c10d \
    #     --rdzv_id $CLOUD_ML_JOB_ID \
    #     --rdzv_endpoint=$(if [[ $RANK -gt 0 ]]; then echo $MASTER_ADDR;else echo localhost;fi):$MASTER_PORT \
    #     NemoHossein/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    #     --config-path="/workspace/a3-bandwidth-test/a3-mega/vertex/nemo/nemo-configs" \
    #     --config-name="llama2-7b" \
    #     +trainer.num_nodes="$NNODES" \
    #     +exp_manager.explicit_log_dir="/tmp/nemo-experiments/results" \
    #     +exp_manager.version="$JOB_IDENTIFIER" \
    #     +exp_manager.exp_dir="/tmp/exp" \
    #     +model.data.data_prefix="[]" 
    #  \
    # > /tmp/logs/rank-$NODE_RANK.log 2>&1 &
    # +model.data.index_mapping_dir="/tmp/index_mapping_dir" \
    # ${workload_arguments[@]} \

echo "Launched rank $NODE_RANK with PID $!"
echo "Logs are available at /tmp/logs/rank-$NODE_RANK.log"
TORCH_PIDS[$LOCAL_RANK]=$!

# for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
#     RANK=$((8*$NODE_RANK + $LOCAL_RANK))
#     
#     OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK HYDRA_FULL_ERROR=1 \
#     nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
#     -o /tmp/hosseins-vertex-test/$JOB_IDENTIFIER/rank-$RANK \
#     --session-new "nemo-rank$RANK" \
#     python NemoHossein/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
#     --config-path="/workspace/a3-bandwidth-test/a3-mega/vertex/nemo/nemo-configs" \
#     --config-name="llama2-7b" \
#     +trainer.num_nodes="$NNODES" \
#     +exp_manager.explicit_log_dir="/tmp/nemo-experiments/results" \
#     +exp_manager.version="$JOB_IDENTIFIER" \
#     +exp_manager.exp_dir="/tmp/exp" \
#     +model.data.data_prefix="[1.0,gs://northam-ce-mlai-tpu/wikipedia/hfbpe_gpt_training_data_text_document]" \
#     > /tmp/logs/rank-$RANK.log 2>&1 &
#     # +model.data.index_mapping_dir="/tmp/index_mapping_dir" \
#     # ${workload_arguments[@]} \
# 
#     echo "Launched rank $RANK with PID $!"
#     echo "Logs are available at /tmp/logs/rank-$RANK.log"
#     TORCH_PIDS[$LOCAL_RANK]=$!
# done

if [ "$NODE_RANK" -eq "1" ]; then
    echo "Launching nvidia-smi in daemon mode with (20 sec delay)"
    nvidia-smi dmon -d 20 -s pum &
fi

if [ "$NODE_RANK" -eq "0" ] && { ! [ -z ${EMBEDDED_TENSORBOARD_TARGET} ]; }; then
    echo "Launching an embedded Tensorboard against log directory $EMBEDDED_TENSORBOARD_TARGET"
    tensorboard --logdir $EMBEDDED_TENSORBOARD_TARGET &
    wait # <-- This will indefinitely stall node rank 0
fi

# # Wait for Torch processes (might be problematic if only one fails)
# for PID in ${TORCH_PIDS[*]}; do
#     echo "Waiting on Torch PID $PID"
#     wait $PID
# done
# 
# sleep 600

echo "Pod on $(hostname --fqdn) is exiting"

# copy one of the yaml config files, like llama2... to the helm folder -> selected-congifuration.yaml

# DLL Logger
# flags in the llama2-7B yaml file

#
##
###
####
#####
#####
####
###
##
#
#
##
###
####
#####
#####
####
###
##
#
