#!/usr/bin/env bash
#SBATCH --job-name swin_base_classification
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --time 24:00:00
#SBATCH --qos=normal # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-03.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk

## Preparation
mkdir -p /raid/student.eryber17 # create a folder to hold your data. It's a good idea to use this path pattern: /raid/<subdomain>.<username>.

srun echo copying training data
if [[ ! -d /raid/student.eryber17/swin_classification_data ]]; then
     # Wrap this copy command inside the if condition so that we copy data only if the target folder doesn't exist
     cp -a -r /user/student.aau.dk/eryber17/swin_classification_data /raid/student.eryber17/swin_classification_data
fi
##if [[ ! -d /raid/student.asol17/output ]]; then
#     mkdir -p /raid/student.asol17/output
#fi

srun echo start training
srun singularity run --nv -B /user/student.aau.dk/eryber17/custom_mmdetection_configs:/custom_configs -B /raid/student.eryber17/swin_classification_data:/data -B /user/student.aau.dk/eryber17/work_dirs:/work_dirs /user/student.aau.dk/eryber17/mmclassification.sif python /mmclassification/tools/train.py --work-dir /work_dirs /custom_configs/swin-base_224x224_custom.py
srun echo finished training

#srun --gres=gpu:0 --pty singularity shell -B /user/student.aau.dk/eryber17/custom_mmdetection_configs:/custom_configs -B /user/student.aau.dk/eryber17/swinpoc_data:/mmdetection/data -B /user/student.aau.dk/eryber17/work_dirs:/work_dirs mmdetection.sif
#python /mmdetection/tools/train.py --work-dir /work_dirs /custom_configs/mask_cascade_mask_rcnn_swin_tiny_custom.py