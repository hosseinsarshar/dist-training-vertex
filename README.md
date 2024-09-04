# Multi-node Pre-training, Continued pre-training and supervised fine-tuning on A3-Mega (H100x8) with Vertex
This repo provides examples on how to launch multi-node distributed training on A3-Mega (H100x8) on Vertex

This repo contains:

## Pretraining with Nemo (Pytorch)
Follow these steps to run this example:

- **Dockerfile**: using the provided [Dockerfile](nemo/Dockerfile), build an image from [NVIDIA Nemo image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags).
- **Nemo Config**: Refer to the nemo/configs directory as example implementations for different models. You can either use these or modify them to your liking.
- **Entry script**: The [job.sh](nemo/job.sh) bash script contains the logic to set the required environment variabls for `TCPXO` and the `torchrun` launcher to start the Nemo pretraining job. `NNODES`, `CONFIG_PATH`, `CONFIG_NAME`, and `LOG_DIR` are required for the script to work properly. For continued pretraining and full sft, please read the section on converting checkpoints before starting.
- **Launch script**: The [launch.sh](nemo/launch.sh) script will construct the REST API call used to launch the job. It takes up to 3 positional arguments and will construct the command using the vertex-payload.json file as a template. Recommended configurations will be used depending on the job type you specify when calling launch.sh.
- **Vertex payload**: The [vertex-payload.json](nemo/vertex-payload.json) file to start a job in Vertex. The vertex-payload.json file is dynamically updated when calling the nemo/launch.sh file. Launch.sh will construct the payload.  
```
export REGION=us-central1
export PROJECT_ID=YOUR_PROJECT

./launch.sh pretraining <llama3-70b|llama2-7b> /tmp
```

**More examples for launch.sh:**

```
# create a cluster but don't launch the job, write logs to GCS
./launsh.sh pretraining llama2-7b /gcs/your-gcs-bucket-name --debug

# launch a job with llama3-70b and write logs to GCS
./launch.sh pretraining llama3-70b /gcs/your-gcs-bucket-name 
```
**note**: "--debug" is optional and will create the cluster, clone this repo and run sleep infinity to keep the cluster up. This is useful if you want to ssh into the machines directly, but it will not kick off the actual training job, you will have to do this manually on every node.

To manually run the job, you must do so on every node, the launch.sh script will set the following environment variables for you if running in --debug mode, otherwise, you'll need to set them manually. Please see below for examples (but again, you do not have to set these even when running manually if launching with launch.sh):

```
NNODES=
CONFIG_PATH="/workspace/dist-training-vertex/nemo/llama2-7b/"
CONFIG_NAME="llama2-7b.yaml"
ADDITIONAL_ARGS="++model.micro_batch_size=1 ++trainer.max_steps=2 ++trainer.limit_val_batches=0.0 ++trainer.val_check_interval=1"
export LOG_DIR="/gcs/<bucket-name>/llama2-7b-pretraining"
```
run script:
```
chmod +x ./dist-training-vertex/nemo/job.sh && ./dist-training-vertex/nemo/job.sh
```

## Note on Continual Pretraining or Full SFT with Nemo (Pytorch)

Before running continual-pretraining or full-sft, you must have a nemo checkpoint file accessible on worker 0 locally.

A nemo checkpoint file is a compatible file containing configurations and a model checkpoint that are recognizable by the nemo framework. A nemo file can be created by converting a huggingface checkpoint directory into a nemo file. For steps on converting HF assets, please see below:

### Model Conversion Steps 
- checkpoint conversion
```
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path /path/to/hf/checkpoint --output_path /path/to/nemo-checkpoint   
```

Copy your nemo checkpoint file to a gcs bucket for convenience.

For similar way to convert llama3 model: [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/starcoder2/checkpointconversion.html)

- transfer checkpoint when launching job

Whether running in --debug mode or launching a full job using launch.sh, set GCS_PATH_TO_CKPT environment variable before calling launch.sh. Doing this will trigger a transfer of the nemo checkpoint file from GCS to the worker 0 node under /workspace/converted_models/$MODEL_NAME.nemo". This can take a few minutes.

ex:
```
export GCS_PATH_TO_CKPT=gs://bucket/model.nemo

./launch.sh <continual-pretraining|full-sft> <llama3-70b|llama2-7b> <LOG_DIR> [--debug]
```

- proceed with training as usual

## 2- Stable Diffusion Diffusers with WebDataset (Pytorch)
*Coming soon*

## 3- Pretraining Llama2-7B with MaxText (JAX)
*Coming soon*

