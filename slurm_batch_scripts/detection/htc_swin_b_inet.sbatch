#!/usr/bin/env bash
#SBATCH --job-name htc_swin-b
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --time 72:00:00
#SBATCH --qos=allgpus # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk

## Preparation
mkdir -p /raid/student.eryber17 # create a folder to hold your data. It's a good idea to use this path pattern: /raid/<subdomain>.<username>.

srun echo copying training data
if [[ ! -d /raid/student.eryber17/dataset_splitted ]]; then
     # Wrap this copy command inside the if condition so that we copy data only if the target folder doesn't exist
     cp -a -r /user/student.aau.dk/eryber17/dataset_splitted /raid/student.eryber17/dataset_splitted
fi

if [[ ! -d /user/student.aau.dk/eryber17/htc_swin_b_inet2_work_dir ]]; then
     mkdir -p /user/student.aau.dk/eryber17/htc_swin_b_inet2_work_dir
fi

srun echo start training
srun singularity run --nv -B /user/student.aau.dk/eryber17/custom_mmdetection_configs/detection:/custom_configs -B /raid/student.eryber17/dataset_splitted:/data -B /user/student.aau.dk/eryber17/htc_swin_b_inet2_work_dir:/work_dir -B /user/student.aau.dk/eryber17/checkpoints:/checkpoints /user/student.aau.dk/eryber17/mmdetection.sif python /mmdetection/tools/train.py --work-dir /work_dir /custom_configs/htc_swin_b_inet.py
srun echo finished training
