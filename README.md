# Multi-node Pre-training, Continued pre-training and supervised fine-tuning on A3-Mega (H100x8) with Vertex
This repo provides examples on how to launch multi-node distributed training on A3-Mega (H100x8) on Vertex

This repo contains:

## 1.1 - Pretraining Llama2-7B with Nemo (Pytorch)
Follow these steps to run this example:

- **Dockerfile**: using the provided [Dockerfile](nemo/Dockerfile), build an image from [NVIDIA Nemo image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags).
- **Nemo Config**: Use the [llama2-7b.yaml](nemo/configs/llama2-7b.yaml) Nemo config file as the reference - you can either use it as is or modify it to your liking.
- **Entry script**: The [job.sh](nemo/job.sh) bash script contains the logic to set the required environment variabls for `TCPXO` and the `torchrun` launcher to start the Nemo pretraining job. `NNODES`, `CONFIG_PATH`, `CONFIG_NAME`, and `LOG_DIR` are required for the script to work properly.
- **Vertex payload**: The [llama2-7b-vertex-payload.json](nemo/pretraining/llama2-7b-vertex-payload.json) file to start a job in Vertex. Make sure you set the right value for the `NNODES` environment variable to reflect the right number of nodes participating in your training job. `CONFIG_PATH`, and `CONFIG_NAME` variables are to set the absolute path to the `Nemo` config file, `ADDITIONAL_ARGS` to provide other `hydro` parameters, and `LOG_DIR` to set the path to the log directory, it can be a local path or a path on gcs. Below is an example of values for these parameters:

```
NNODES=4
CONFIG_PATH="/workspace/dist-training-vertex/nemo/llama2-7b/"
CONFIG_NAME="llama2-7b.yaml"
ADDITIONAL_ARGS="++model.micro_batch_size=1 ++trainer.max_steps=2 ++trainer.limit_val_batches=0.0 ++trainer.val_check_interval=1"
export LOG_DIR="/gcs/<bucket-name>/llama2-7b-pretraining"
```

The right parameters are set in both json payload files: [Llama2-7B](nemo/pretraining/llama2-7b-vertex-payload.json) and [Llama3-70B](nemo/pretraining/llama3-70b-vertex-payload.json).
- **Submit the job**: Use the following curl command to kick off the job on Vertex:

```
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @nemo/pretraining/llama3-70b-vertex-payload.json \
     "https://<reigon>-aiplatform.googleapis.com/v1/projects/<project-id>/locations/<reigon>/customJobs"
```
## 1.2 - Pretraining Llama3-70B with Nemo (Pytorch)
The same steps as 1.1 apply (except please refer to nemo/llama3-70b directory). Use the following command to kick off the job:

```
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @nemo/llama3-70b/vertex-payload.json \
     "https://<reigon>-aiplatform.googleapis.com/v1/projects/<project-id>/locations/<reigon>/customJobs"
```
## 1.3 - Continual Pretraining with llama3-70B with Nemo (Pytorch)
- checkpoint conversion

python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path /path/to/hf/checkpoint --output_path /local/directory   

- import checkpoint

In llamax-xxb.yaml file, update restore_from_ckpt: /path/to/nemo-checkpoint

- continue pretraining in the same way as 1.2

Note: you may want to enable checkpointing in the configuration file as well.
## 1.4 - Full Supervised Fine-tuning with llama3-70B with Nemo (Pyotch)


## 2- Stable Diffusion Diffusers with WebDataset (Pytorch)
*Coming soon*

## 3- Pretraining Llama2-7B with MaxText (JAX)
*Coming soon*

