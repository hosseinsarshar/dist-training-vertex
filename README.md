# Distributed Training on A3-Mega (H100x8) with Vertex
This repo provides examples on how to launch distributed training on A3-Mega (H100x8) on Vertex

This repo contains:

## 1.1 - Pretraining Llama2-7B with Nemo (Pytorch)
Follow these steps to run this example:

- **Dockerfile**: using the provided [Dockerfile](nemo/Dockerfile), build an image from [NVIDIA Nemo image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags).
- **Nemo Config**: Use the [llama2-7b.yaml](nemo/llama2-7b/llama2-7b.yaml) Nemo config file as the reference - you can either use it as is or modify it to your liking.
- **Entry script**: The [job.sh](nemo/job.sh) bash script contains the logic to set the required environment variabls for `TCPXO` and the `torchrun` launcher to start the Nemo pretraining job. `NNODES`, `CONFIG_PATH`, and `CONFIG_NAME` are required for the script to work properly.
- **Vertex payload**: The [vertex-payload.json](nemo/llama2-7b/vertex-payload.json) file to start a job in Vertex. Make sure you set the right value for the `NNODES` environment variable to reflect the right number of nodes participating in your training job. `CONFIG_PATH`, and `CONFIG_NAME` variables are to set the absolute path to the `Nemo` config file and `ADDITIONAL_ARGS` to provide other `hydro` parameters. Below is an example of values for these parameters:
```
NNODES=4
CONFIG_PATH="/workspace/dist-training-vertex/nemo/llama2-7b/"
CONFIG_NAME="llama2-7b.yaml"
ADDITIONAL_ARGS='+exp_manager.explicit_log_dir="/tmp/nemo-experiments/results" +exp_manager.exp_dir="/tmp/exp" ++model.micro_batch_size=1 ++trainer.max_steps=4 +model.data.data_prefix="[]" ++trainer.limit_val_batches=0.0 ++trainer.val_check_interval=1 ++model.micro_batch_size=2'
```
- **Submit the job**: Use the following curl command to kick off the job on Vertex:

```
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @nemo/llama2-7b/vertex-payload.json \
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

## 2- Stable Diffusion Diffusers with WebDataset (Pytorch)
*Coming soon*

## 3- Pretraining Llama2-7B with MaxText (JAX)
*Coming soon*

