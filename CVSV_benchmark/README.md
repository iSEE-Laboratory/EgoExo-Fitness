# Cross-View Sequence Verification (CVSV) Benchmark

## News
- [2024.06.25] CVSV Baseline Code is available.
- [2024.06.25] Configs on experiments of various training sources (i.e., Table 4 in our paper) are available. 

## Environment
You can build the Anaconda environment via the following script.
```
conda env create -f environment.yml
```

## Usage
Please refer to **`run.sh`** to see the running scripts. Our experiments can be conducted on NVIDIA RTX 3090Ti GPUs. 

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
The authors thank the team members of WeakSVR for providing the code used in her works and useful feedback.
```
@inproceedings{dong2023weakly,
  title={Weakly supervised video representation learning with unaligned text for sequential videos},
  author={Dong, Sixun and Hu, Huazhang and Lian, Dongze and Luo, Weixin and Qian, Yicheng and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2437--2447},
  year={2023}
}
```
