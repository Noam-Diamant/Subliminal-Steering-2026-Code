#!/bin/bash -l
#SBATCH --job-name=finetune_job_%j     
#SBATCH --output=finetune_job_%j.out        
#SBATCH --error=finetune_job_%j.err         
#SBATCH --partition=H200-12h         
#SBATCH --gres=gpu:1                   
#SBATCH --mem=80G  


# activate conda
cd ..
source venv/bin/activate
cd scripts

echo "Starting finetuning job on node $SLURM_NODELIST"

#./run15_local.sh --teacher Qwen25-7B --student Llama-32-3B
# ./run15_local.sh --teacher Llama-32-3B --student Qwen25-7B --trial
# ./run15_local.sh --teacher Llama-32-3B --student Qwen25-7B --trial

# ./run15_local.sh --mode prompted
#./run15_local.sh --mode prompted --teacher Qwen25-7B --student Llama-32-3B

#./run15_local.sh --mode prompted

#./run15_local.sh --mode prompted --models Llama-32-3B

./run15_local.sh --mode prompted --teacher Llama-32-3B --student Qwen25-7B


echo "Job finished."