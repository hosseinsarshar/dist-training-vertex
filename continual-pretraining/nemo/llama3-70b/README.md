## Model Conversion Steps 
- checkpoint conversion

python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path /path/to/hf/checkpoint --output_path /local/directory   

- import checkpoint

In llamax-xxb.yaml file, update restore_from_ckpt: /path/to/nemo-checkpoint

- continue pretraining in the same way as pretraining.

Note: you may want to enable checkpointing in the configuration file as well.

For similar way to convert llama3 model: [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/starcoder2/checkpointconversion.html)

