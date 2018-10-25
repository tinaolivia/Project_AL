#!/bin/sh
#SBATCH --partition=EPIC2
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH --job-name="cnn"
#SBATCH --output=run.out

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR} 
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load GCC/7.3.0-2.30 OpenMPI/3.1.1
module load icc/2018.3.222-GCC-7.3.0-2.30 impi/2018.3.222
module load ifort/2018.3.222-GCC-7.3.0-2.30 impi/2018.3.222
module load Python/3.6.6

source activate ALenv

python3 main.py 