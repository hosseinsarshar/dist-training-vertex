# Multi-node Pre-training, Continued pre-training and supervised fine-tuning on A3-Mega (H100x8) with Vertex
This repo provides examples on how to launch multi-node distributed training on A3-Mega (H100x8) on Vertex

This repo contains:

## 1.1 - Pretraining Llama2-7B with Nemo (Pytorch)
Follow these steps to run this example:

- **Dockerfile**: using the provided [Dockerfile](nemo/Dockerfile), build an image from [NVIDIA Nemo image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags).
- **Nemo Config**: Use the [llama2-7b.yaml](nemo/configs/llama2-7b.yaml) Nemo config file as the reference - you can either use it as is or modify it to your liking.
- **Entry script**: The [job.sh](nemo/job.sh) bash script contains the logic to set the required environment variabls for `TCPXO` and the `torchrun` launcher to start the Nemo pretraining job. `NNODES`, `CONFIG_PATH`, `CONFIG_NAME`, and `LOG_DIR` are required for the script to work properly.
- **Vertex payload**: The [vertex-payload.json](nemo/vertex-payload.json) file to start a job in Vertex. The vertex-payload.json file is dynamically updated when calling the nemo/launch.sh file. Use it to construct the payload. Recommended configurations will be used depending on the job type you specify when calling launch.sh. Below is an example:

```
Usage: launch.sh <TRAIN_TYPE> <MODEL_NAME> <LOG_DIR> [--debug]

ex:
export REGION=us-central1
export PROJECT_ID=YOUR_PROJECT

1. ./launsh.sh continual-pretraining llama3-70b /tmp --debug
2. ./launsh.sh pretraining llama2-7b /gcs/your-gcs-bucket-name
3. ./launch.sh full-sft llama3-70b /gcs/your-gcs-bucket-name 
```
**note**: debug is optional and will create the cluster, clone this repo and run sleep infinity. This is useful if you want to ssh into the machines directly, but it will not kick off the actual training job, you will have to do this manually on every node.

To manually run the job, on every node, make sure to set the following environment variables:

```
NNODES=4
CONFIG_PATH="/workspace/dist-training-vertex/nemo/llama2-7b/"
CONFIG_NAME="llama2-7b.yaml"
ADDITIONAL_ARGS="++model.micro_batch_size=1 ++trainer.max_steps=2 ++trainer.limit_val_batches=0.0 ++trainer.val_check_interval=1"
export LOG_DIR="/gcs/<bucket-name>/llama2-7b-pretraining"
```
run script:
```
chmod +x ./dist-training-vertex/nemo/job.sh && ./dist-training-vertex/nemo/job.sh
```

## Note on Continual Pretraining with llama3-70B with Nemo (Pytorch)

Before running continual-pretraining, you must have a nemo checkpoint file accessible on worker 0 locally.

A nemo checkpoint file is a compatible file containing configurations and a model checkpoint that are recognizable by the nemo framework. A nemo file can be created by converting a huggingface checkpoint directory into a nemo file. For steps on converting HF assets, please see below:

### Model Conversion Steps 
- checkpoint conversion

python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path /path/to/hf/checkpoint --output_path /path/to/nemo-checkpoint   

- import checkpoint

In **vertex-payload.json file, overwrite the `restore_from_ckpt` parameter by adding `++resume_from_checkpoint=/path/to/nemo-checkpoint/llama-3-70b.nemo"` to the `ADDITIONAL_ARGS` env variable.

- continue pretraining in the same way as pretraining.

Note: you may want to enable checkpointing in the configuration file as well.

For similar way to convert llama3 model: [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/starcoder2/checkpointconversion.html)


## 1.4 - Full Supervised Fine-tuning with llama3-70B with Nemo (Pyotch)


## 2- Stable Diffusion Diffusers with WebDataset (Pytorch)
*Coming soon*

## 3- Pretraining Llama2-7B with MaxText (JAX)
*Coming soon*

