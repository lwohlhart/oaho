#!/usr/bin/env bash
##########################################################
# where to write tfevents
OUTPUT_DIR="out"
# experiment settings
TRAIN_BATCH=6 #4 # 8
EVAL_BATCH=6 #4 # 8
BATCH=6 #4 # 8

LR=0.0001
EPOCHS=1000
# create a job name for the this run
prefix="oaho"
now=$(date +"%Y%m%d_%H_%M_%S")
JOB_NAME="$prefix"_"$now"
# locations locally or on the cloud for your files
TRAIN_FILES="data/oaho_synth_train*.tfrecord"
EVAL_FILES="data/oaho_synth_val*.tfrecord"
TEST_FILES="data/oaho_synth_test*.tfrecord"

WARM_START_DIR="out/oaho_20190708_17_03_11/"

MODEL_DEF="config/model.yaml"
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
JOB_DIR=${OUTPUT_DIR}/${JOB_NAME}

###################
# Add notes to the log file based on the current information about this training job close vim to start training
# useful if you are running lots of different experiments and you forget what values you used
echo "---
## ${JOB_NAME}" >> training_log.md
echo "Learning Rate: ${LR}" >> training_log.md
echo "Epochs: ${EPOCHS}" >> training_log.md
echo "Batch Size (train/eval): ${TRAIN_BATCH}/ ${EVAL_BATCH}" >> training_log.md
echo "### Hypothesis
" >> training_log.md
echo "### Results
" >> training_log.md
#vim + training_log.md
###################

# activate the virtual environment
if [[ -z $2 ]]; then
    set +u
    source $ENV_NAME/bin/activate
    set -u
fi

# start training
CUDA_VISIBLE_DEVICES="$GPU_ID"
mprof run ipython -i initialisers/task.py -- \
        --job-dir ${JOB_DIR} \
        --train-batch-size ${BATCH} \
        --eval-batch-size ${BATCH} \
        --learning-rate ${LR} \
        --num-epochs ${EPOCHS} \
        --train-files ${TRAIN_FILES} \
        --eval-files ${EVAL_FILES} \
        --test-files ${TEST_FILES} \
        --export-path "${OUTPUT_DIR}/exports" \
	--warm-start-dir "${WARM_START_DIR}" \
        --model-def "${MODEL_DEF}"
#&>runlogs/$JOB_NAME.log & echo "$!" > runlogs/$JOB_NAME.pid

echo "Job launched."
