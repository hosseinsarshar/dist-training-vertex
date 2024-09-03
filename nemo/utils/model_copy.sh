#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: utils\model_copy.sh <GCS_PATH_TO_CKPT> <CONVERTED_MODEL_PATH> <LOG_DIR>"
    echo
    echo "  <GCS_PATH_TO_CKPT>     Location of the model checkpoint (in '.nemo' format)"
    echo "  <CONVERTED_MODEL_PATH> Local path on the A3-Mega node to copy the checkpoint to"
    echo "  <LOG_DIR>              Path to the bucket (/gcs/NAME_OF_BUCKET) - the bucket should be ideally in the same region"
    exit 1
}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

COPY_COMPLETE_PATH="$LOG_DIR/$CLOUD_ML_JOB_ID/copy-complete.txt"
if [ "$RANK" -eq 0 ]; then
    echo "Copying the checkpoint from $1 to $2"
    gcloud storage cp "$1" "$2"
    
    # Ensure the directory exists
    mkdir -p "$(dirname "$COPY_COMPLETE_PATH")"  
    
    # Create the copy-complete.txt file
    touch "$COPY_COMPLETE_PATH"
    echo "The checkpoint is copied on RANK=$RANK."
else
    echo "On RANK $RANK, waiting for copy-complete.txt to be created at $COPY_COMPLETE_PATH ..."
    
    # Repeatedly check if the file exists
    while [ ! -f "$COPY_COMPLETE_PATH" ]; do
        sleep 5  # Wait for 5 seconds before checking again
    done

    echo "$COPY_COMPLETE_PATH found on RANK=$RANK. Resuming the operation."
    # Place the rest of your script here
fi