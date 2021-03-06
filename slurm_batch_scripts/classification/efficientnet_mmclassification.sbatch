#!/usr/bin/env bash
#SBATCH --job-name efficientnet_b0
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --time 48:00:00
#SBATCH --qos=allgpus # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk

## Preparation
mkdir -p /raid/student.eryber17 # create a folder to hold your data. It's a good idea to use this path pattern: /raid/<subdomain>.<username>.

srun echo copying training data
if [[ ! -d /raid/student.eryber17/classification_data ]]; then
     # Wrap this copy command inside the if condition so that we copy data only if the target folder doesn't exist
     cp -a -r /user/student.aau.dk/eryber17/classification_data /raid/student.eryber17/classification_data
fi

if [[ ! -d /user/student.aau.dk/eryber17/efficientnet_b0_work_dir ]]; then
     mkdir -p /user/student.aau.dk/eryber17/efficientnet_b0_work_dir
fi

srun echo start training
srun singularity run --nv -B /user/student.aau.dk/eryber17/custom_mmdetection_configs/classification:/custom_configs -B /raid/student.eryber17/classification_data:/data -B /user/student.aau.dk/eryber17/efficientnet_b0_work_dir:/work_dir /user/student.aau.dk/eryber17/mmclassification.sif python /mmclassification/tools/train.py --work-dir /work_dir /custom_configs/efficientnet_224x224_custom.py
srun echo finished training
