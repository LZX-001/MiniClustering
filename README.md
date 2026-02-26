It is the code for MiniClustering

## Training BYOL
```bash
torchrun  --nproc_per_node=1 --master_port 2000 main_byol.py
```
You may choose a save_dir to save checkpoints and logs. 

And you can adjust the experimental setup and hyperparameters in the main_byol.py.

## Training MiniClustering
```bash
torchrun  --nproc_per_node=1 --master_port 2000 MiniClustering.py
```
use the specific save_dir and checkpoint for MiniClustering.