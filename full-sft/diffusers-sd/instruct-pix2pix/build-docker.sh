# Dockerfile create
# Build from https://github.com/huggingface/diffusers/blob/main/docker/diffusers-pytorch-xformers-cuda/Dockerfile
export PIX2PIX_FOLDER=instruct_pix2pix
mkdir $PIX2PIX_FOLDER
sudo chmod +777 -R $PIX2PIX_FOLDER
cd $PIX2PIX_FOLDER
wget https://raw.githubusercontent.com/huggingface/diffusers/main/docker/diffusers-pytorch-xformers-cuda/Dockerfile

# Update the Dockerfile to add diffusers and install the requirements for instruct_pix2pix, right before the CMD ["/bin/bash"]
nano Dockerfile
RUN pip install git+https://github.com/huggingface/diffusers
RUN git clone https://github.com/huggingface/diffusers.git
RUN pip install -r https://raw.githubusercontent.com/huggingface/diffusers/main/examples/instruct_pix2pix/requirements.txt

chmod +x -R diffusers

PROJECT_ID="<YOUR-PROJECT-ID>"  # @param {type:"string"}
REPOSITORY="diffusers-sd-training-repository"

image_accelerate_train="sd-pix2pix_train"
REGION="<YOUR-REGION>"
hostname="${REGION}-docker.pkg.dev"
tag="latest"

SD_PIX2PIX_TRAIN_DOCKER_URI="${hostname}/${PROJECT_ID}/${REPOSITORY}/${image_accelerate_train}:${tag}"

gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

gcloud artifacts repositories create $REPOSITORY --repository-format=docker \
--location=$REGION --description="Stable Diffusion A3 Mega training repository"

docker build -t $SD_PIX2PIX_TRAIN_DOCKER_URI -f Dockerfile .
docker push $SD_PIX2PIX_TRAIN_DOCKER_URI