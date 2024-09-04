#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: utils\model_copy.sh <GCS_PATH_TO_CKPT> <CONVERTED_MODEL_PATH>"
    echo
    echo "  <GCS_PATH_TO_CKPT>     Location of the model checkpoint (in '.nemo' format)"
    echo "  <CONVERTED_MODEL_PATH> Local path on the A3-Mega node to copy the checkpoint to"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

if [ "$RANK" -eq 0 ]; then
    echo "Copying the checkpoint from $1 to $2"
else
    echo "On RANK $RANK, waiting for the copy on RANK 0 to complete ..."
fi

torchrun  --nproc_per_node=1 \
    --nnodes=${NNODES} \
    --rdzv-backend=static \
    --node_rank=$RANK \
    --rdzv_id $CLOUD_ML_JOB_ID \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    dist-training-vertex/nemo/utils/model_copy.py --src $1 --dest $2

echo "RANK=$RANK - Copying the checkpoint from [$1] to [$2] completed. Continuing to the next step of the training."
