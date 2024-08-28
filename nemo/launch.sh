#!/bin/bash

#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage function to display help
usage() {
    echo "Usage: launch.sh <TRAIN_TYPE> <MODEL_NAME> <LOG_DIR> [--debug]"
    echo
    echo "  <TRAIN_TYPE>     Job type (options: pretraining,continual-pretraining,full-sft)"
    echo "  <MODEL_NAME>     Model name (options: llama2-7b,llama3-70b)"
    echo "  <LOG_DIR>        Path to bucket (/gcs/NAME_OF_BUCKET) or Relative or absolute path to the log directory (/tmp)"
    echo "  --debug          Pass sleep infinity to launch command"
    exit 1
}
export TRAIN_TYPE=$1
export MODEL_NAME=$2
export LOG_DIR=$3

if [ $4 = "--debug" ]; then
    export DEBUG=$4
fi

echo JOB_TYPE:$TRAIN_TYPE
echo MODEL_NAME:$MODEL_NAME
echo LOG_DIR:$LOG_DIR
echo DEBUG:$DEBUG

if [ -z "$TRAIN_TYPE" ] || [ -z "$MODEL_NAME" ] || [ -z "$LOG_DIR" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

# == construct job launch command == 

# create base job launch command 
export LAUNCH_CMD="git clone https://github.com/hosseinsarshar/dist-training-vertex.git &&"

# add checkpoint transfer to launch command # NOTE: set BUCKET env var before calling launch.sh
if [ $TRAIN_TYPE = "continual-pretraining" ] || [ $TRAIN_TYPE = "continual-pretraining" ]; then
    export CONVERTED_MODEL_PATH="/workspace/converted_models/$MODEL_NAME.nemo"
    export LAUNCH_CMD="$LAUNCH_CMD gsutil -m cp $GCS_PATH_TO_CKPT $CONVERTED_MODEL_PATH &&"
    export ADDITIONAL_ARGS="$ADDITIONAL_ARGS ++model.resume_from_checkpoint=$CONVERTED_MODEL_PATH"
fi

# if in debug mode add sleep infinity to launch command
if [ -z "$DEBUG" ]; then
    export LAUNCH_CMD="$LAUNCH_CMD chmod +x ./dist-training-vertex/nemo/job.sh && ./dist-training-vertex/nemo/job.sh"
else 
    export LAUNCH_CMD="$LAUNCH_CMD sleep infinity"
fi

# == set job specific parameters based on model ==
if [ $MODEL_NAME = 'llama3-70b' ]; then
    export NNODES=8
    export MICRO_BATCH=2
elif [ $MODEL_NAME = 'llama2-7b' ]; then
    export NNODES=4
    export MICRO_BATCH=1
fi

export REPLICA_COUNT=$(($NNODES-1))

# == define additional args ==
export ADDITIONAL_ARGS="++model.micro_batch_size=$MICRO_BATCH ++trainer.max_steps=2 ++trainer.limit_val_batches=0.0 ++trainer.val_check_interval=1"

# == create json stucture with existing environment variables ==
json_job=$(envsubst < vertex-payload.json)

curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d $json_job \
     "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/customJobs"

