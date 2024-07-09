# Cross-View Sequence Verification (CVSV) Benchmark

## News
- [2024.07.09] CVSV Baseline Code and Annotations are available.

## Data Preparation
Please follow the [instructions](../README.md) to apply for the link to download the dataset.

## Environment
You can build the Anaconda environment via the following script.
```
conda env create -f environment.yml
```

## Usage
Please refer to **`run.sh`** to see the running scripts. An example is shown as below:
```
CUDA_VISIBLE_DEVICES=4,5 python train.py --config configs/All_Pairs/train_config_egofbau_CAT_pair_ClsSeq.yml --pair
```
Our experiments can be conducted on NVIDIA RTX 3090Ti GPUs. 

## Citation
Please cite it if you find this work useful.
```
@article{li2024egoexo,
  title={EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding},
  author={Li, Yuan-Ming and Huang, Wei-Jin and Wang, An-Lan and Zeng, Ling-An and Meng, Jing-Ke and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2406.08877},
  year={2024}
}
```

## Acknowledgement
The authors thank the team members of [WeakSVR](https://github.com/svip-lab/WeakSVR/tree/main) for providing the code used in her works and useful feedback.
