#!/bin/bash

#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage function to display help
: "${GPUS_PER_NODE:?Must set GPUS_PER_NODE}"



usage() {
    echo "Usage: $0 --nnodes <NNODES> --config_path <CONFIG_PATH> --config_name <CONFIG_NAME> --log_bucket_dir <LOG_DIR>  [other options...]"
    echo
    echo "Mandatory options:"
    echo "  --nnodes <NNODES>                     Number of nodes (also sets +trainer.num_nodes)."
    echo "  --config_path <CONFIG_PATH>           Path to config."
    echo "  --config_name <CONFIG_NAME>           Name of config file such as 'llama2-7b.yaml'."
    echo "  --log_dir <LOG_DIR>                   Relative or absolute path to the log directory - such as /tmp/ or /gcs/<bucket-name>"
    echo
    echo "  Set ADDITIONAL_ARGS environment variable for hydra extra parameters."
    echo
    exit 1
}

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
export LD_LIBRARY_PATH=${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH="/usr/local/nccl-plugin/lib64:/usr/local/nvidia/lib64/:${LD_LIBRARY_PATH}"
echo "Warning: Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH to override the NCCL library"

ldconfig /usr/local/nvidia/lib64/
echo "Added /usr/local/nvidia/lib64/ to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'

## To turn on for debugging
# export TORCH_CPP_LOG_LEVEL=INFO # this is to turn on the verbose torch logs
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export NCCL_DEBUG=INFO

echo "Downloading GPT vocabulary files"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json &&\
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p ${LOG_DIR}/logs/
mkdir -p ${LOG_DIR}/exp/
mkdir -p ${LOG_DIR}/nemo-experiments/results
mkdir -p ${LOG_DIR}/index_mapping_dir

export GPUS_PER_NODE=8
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
export DYNAMIC_ARGS="+trainer.num_nodes=${NNODES} +exp_manager.explicit_log_dir=\"${LOG_DIR}/nemo-experiments/results\" +model.data.data_prefix=\"[]\" +exp_manager.exp_dir=\"${LOG_DIR}/exp/\" ${ADDITIONAL_ARGS}"

echo RANK:$RANK
echo GPUS_PER_NODE:$GPUS_PER_NODE
echo WORLD_SIZE:$WORLD_SIZE
echo MASTER_PORT:$MASTER_PORT
echo NNODES:$NNODES
echo DYNAMIC_ARGS:$DYNAMIC_ARGS
echo CONFIG_PATH:$CONFIG_PATH
echo CONFIG_NAME:$CONFIG_NAME
echo LOG_DIR:$LOG_DIR

if [ -z "$NNODES" ] || [ -z "$CONFIG_PATH" ] || [ -z "$CONFIG_NAME" ] || [ -z "$LOG_DIR" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

echo "sleep for 10 seconds to let services boot up"
sleep 10

echo "Launching Torch distributed as node rank $RANK out of $NNODES nodes"
OMP_NUM_THREADS=12 RANK=$RANK HYDRA_FULL_ERROR=1 \
torchrun  --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NNODES} \
    --rdzv-backend=static \
    --node_rank=$RANK \
    --rdzv_id $CLOUD_ML_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    $DYNAMIC_ARGS

echo "Training completed on node rank $RANK out of $NNODES nodes"
