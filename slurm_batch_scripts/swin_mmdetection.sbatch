#!/usr/bin/env bash
#SBATCH --job-name mmdetection_test
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --time 6:00:00
#SBATCH --qos=normal # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-03.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asol17@student.aau.dk

## Preparation
#mkdir -p /raid/student.asol17 # create a folder to hold your data. It's a good idea to use this path pattern: /raid/<subdomain>.<username>.

#if [[ ! -d /raid/student.asol17/cascadedata ]]; then
     # Wrap this copy command inside the if condition so that we copy data only if the target folder doesn't exist
#     cp -a /user/student.aau.dk/asol17/instanceSegmentation/data /raid/student.asol17/cascadedata
#fi
##if [[ ! -d /raid/student.asol17/output ]]; then
#     mkdir -p /raid/student.asol17/output
#fi

srun echo start training
srun singularity run --nv -B /user/student.aau.dk/eryber17/custom_mmdetection_configs:/custom_configs -B /user/student.aau.dk/eryber17/swinpoc_data:/user/student.aau.dk/eryber17/data -B /user/student.aau.dk/eryber17/work_dirs:/work_dirs /user/student.aau.dk/eryber17/mmdetection.sif python /mmdetection/tools/train.py --work-dir /work_dirs /custom_configs/mask_cascade_mask_rcnn_swin_tiny_custom.py
srun echo finished training

#srun --gres=gpu:0 --pty singularity shell -B /user/student.aau.dk/eryber17/custom_mmdetection_configs:/custom_configs -B /user/student.aau.dk/eryber17/swinpoc_data:/mmdetection/data -B /user/student.aau.dk/eryber17/work_dirs:/work_dirs mmdetection.sif
#python /mmdetection/tools/train.py --work-dir /work_dirs /custom_configs/mask_cascade_mask_rcnn_swin_tiny_custom.py