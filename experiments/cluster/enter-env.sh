module purge                               # Removes all modules still loaded
module load rhel7/default-gpu              # REQUIRED - loads the basic environment
module unload cuda/8.0
module load cuda/10.1
module load cudnn/7.6_cuda-10.1
module load python/3.7
. "$HOME/Programacio/cnn-limits/venv/bin/activate"
