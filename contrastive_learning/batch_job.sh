#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=10G
#SBATCH -t 0:45:00
#SBATCH -J 9_5
#SBATCH -o 9_5.out
#SBATCH -e 9_5.out
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

##module load anaconda
##source /users/anagara8/data/anagara8/mae/bin/activate
##conda activate /users/akarkada/data/akarkada/conda/envs/py39

#module load python/3.9.0
#module load cuda/11.7.1
#module load gcc/10.2
#module load libjpeg-turbo/2.0.2 
#module load opencv/3.4.1
#. /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
#. /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
#. /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
#conda activate
#. /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh

if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/oscar/runtime/software/external/miniconda3/23.11.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh" ]; then
        . "/oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh"
    else
        export PATH="/oscar/runtime/software/external/miniconda3/23.11.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module load cuda
conda activate dl_project

python -u train_model.py 0.9_5