# Ego2Allo-VLM

Load and save datasets/models to `/ocean/projects/cis250208p/shared`.

## Setup

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc

conda create -n ego2allo python=3.10
conda activate ego2allo
pip install -r requirements.txt

module load cuda/12.6.1
echo $CUDA_HOME
```