#!/bin/bash
#SBATCH --job-name=headmodel_gen
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=2-40
#SBATCH --output=logs/charm_%A_%a.out
#SBATCH --error=logs/charm_%A_%a.err

PROJECT_DIR="/lustre/home/rhadimanilab/nibs_comp"
SUB_ID=$(printf "sub-%03d" ${SLURM_ARRAY_TASK_ID})

T1W_FILE="${PROJECT_DIR}/data/${SUB_ID}/anat/T1w_acpc_dc.nii.gz"
T2W_FILE="${PROJECT_DIR}/data/${SUB_ID}/anat/T2w_acpc_dc.nii.gz"
SETTINGS_FILE="${PROJECT_DIR}/data/derivatives/simnibs_charm/charm.ini"
OUTPUT_DIR="${PROJECT_DIR}/data/derivatives/simnibs_charm/${SUB_ID}"
WORK_DIR="/tmp/nibs_comp/headmodel_gen/${SUB_ID}"

echo "Processing Subject: ${SUB_ID}"
echo "Input T1w file: ${T1W_FILE}"
echo "Input T2w file: ${T2W_FILE}"

rm -rf ${WORK_DIR}
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo "Working Directory: $(pwd)"
echo "Executing: charm ${SUB_ID} ${T1W_FILE} ${T2W_FILE} --usesettings ${SETTINGS_FILE}"

charm ${SUB_ID} ${T1W_FILE} ${T2W_FILE} --usesettings "${SETTINGS_FILE}"

echo "Copying results to ${OUTPUT_DIR}"
mkdir -p ${OUTPUT_DIR}
rsync -av ${WORK_DIR}/ ${OUTPUT_DIR}/

echo "Cleaning up working directory"
rm -rf ${WORK_DIR}

echo "---"
echo "Successfully finished processing ${SUB_ID}"
