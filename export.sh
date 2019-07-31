#!/usr/bin/env bash
##########################################################
# where to write tfevents
OUTPUT_DIR="out"
# experiment settings

# create a job name for the this run
MODEL="model_resnet_deconv"
JOB_DIR="out/oaho_model_resnet_deconv_20190730_17_36_10/"
MODEL_DEF="config/$MODEL.yaml"

##########################################################

if [[ -z $1 && -z $2 ]]; then
    echo "Incorrect arguments specified."
    echo ""
    echo "Usage: ./train_local_single.sh <GPU_ID> [ENV_NAME]"
    echo ""
    exit 1
else
    GPU_ID=$1
    if [[ -z $2 ]]; then
        ENV_NAME="default"
    else
        ENV_NAME=$2
    fi
fi

if [[ -z $LD_LIBRARY_PATH || -z $CUDA_HOME  ]]; then
    echo ""
    echo "CUDA environment variables not set."
    echo "Consider adding them to your shell-rc."
    echo ""
    echo "Example:"
    echo "----------------------------------------------------------"
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"'
    echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"'
    echo 'CUDA_HOME="/usr/local/cuda"'
    echo ""
fi

# needed to use virtualenvs
set -euo pipefail

# get current working directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# create folders if they don't exist of logs and outputs
mkdir -p $DIR/runlogs

# create a local job directory for checkpoints etc


###################
# Add notes to the log file based on the current information about this training job close vim to start training
# useful if you are running lots of different experiments and you forget what values you used
# echo "---
# ## ${JOB_NAME}" >> training_log.md
# echo "Learning Rate: ${LR}" >> training_log.md
# echo "Epochs: ${EPOCHS}" >> training_log.md
# echo "Batch Size (train/eval): ${TRAIN_BATCH}/ ${EVAL_BATCH}" >> training_log.md
# echo "### Hypothesis
# " >> training_log.md
# echo "### Results
# " >> training_log.md
#vim + training_log.md
###################

# # activate the virtual environment
# if [[ -z $2 ]]; then
#     set +u
#     source $ENV_NAME/bin/activate
#     set -u
# fi

# start training
CUDA_VISIBLE_DEVICES="$GPU_ID"
python3 -m initialisers.export \
        --export_path "${OUTPUT_DIR}/exports" \
        --job_dir "${JOB_DIR}" \
        --model_def "${MODEL_DEF}"

echo "Exported."
