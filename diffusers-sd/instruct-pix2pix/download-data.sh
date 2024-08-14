# download the model files
pip install -U "huggingface_hub[cli]"

huggingface-cli login
python3
from huggingface_hub import snapshot_download
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATASET_ID="fusing/instructpix2pix-1000-samples"
LOCAL_DIR="<YOUR-LOCAL-CACHE-DIRECTORY>/instruct_pix2pix_files"
snapshot_download(f"{MODEL_NAME}",local_dir=f"{LOCAL_DIR}/models--runwayml--stable-diffusion-v1-5")
snapshot_download(f"{DATASET_ID}",local_dir=f"{LOCAL_DIR}/datasets--fusing--instructpix2pix-1000-samples", repo_type="dataset")

gcloud storage cp -r instruct_pix2pix_files gs://<YOUR-BUCKET-NAME>