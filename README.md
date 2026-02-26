It is the code for MiniClustering

#### Training BYOL
```bash
torchrun  --nproc_per_node=1 --master_port 2000 main_byol.py
```
You may choose a save_dir to save checkpoints and logs. 

And you can adjust the experimental setup and hyperparameters in the main_byol.py.

#### Training MiniClustering
```bash
torchrun  --nproc_per_node=1 --master_port 2000 MiniClustering.py
```

use the specific save_dir and checkpoint for MiniClustering.

## Acknowlegement
torch_clustering: https://github.com/Hzzone/torch_clustering

## Citation
If this code is helpful, you are welcome to cite our paper.

```
@inproceedings{
li2026minicluster,
title={Mini-cluster Guided Long-tailed Deep Clustering},
author={Zhixin Li and Yuheng Jia and Guanliang Chen and Hui LIU and Junhui Hou},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=3JlljaiQwR}
}
```
