#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=10G
#SBATCH -t 8:00:00
#SBATCH -J second_half
#SBATCH -o second_half.out
#SBATCH -e second_half.out
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

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

python -u train_model.py