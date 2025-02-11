# <p align="center">EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding</p>

### <p align="center">*Yuanming Li, Weijin Huang, Anlan Wang, Lingan Zeng, Jingke Meng, Weishi Zheng*</p>

#### <p align="center">[[Paper w/ Appendix]](https://arxiv.org/abs/2406.08877) </p>

Official repository of ECCV-2024 paper "EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding"

All resources will be available soon. **If you have any question, please feel free to contact us**.

**Contact:** 
- Email: yuanmingli527@gmail.com / liym266@mail2.sysu.edu.cn

## 💬 News
- [2024.06.14] The preprint paper is available.
- [2024.07.02] This work is accepted by ECCV-2024. Many thanks to the co-workers!🥳🎉🎊
- [2024.07.09] EgoExo-Fitness dataset and the raw annotations are open to apply for.🔥🔥🔥 Click [here](./Raw_annotations/README.md) for more details.
- [2024.07.09] Code for ***Cross-View Sequence Verification*** benchmark is available. Click [here](./CVSV_benchmark/README.md) for more details.
- [2025.02.11] EgoExo-Fitness dataset is available on Huggingface. Click [here](https://huggingface.co/datasets/Lymann/EgoExo-Fitness) for more details.


## :paperclip: Abstract
We present **EgoExo-Fitness**, a new full-body action understanding dataset, featuring fitness sequence videos recorded from synchronized egocentric and fixed exocentric (third-person) cameras. Compared with existing full-body action understanding datasets, EgoExo-Fitness not only contains videos from first-person perspectives, but also provides rich annotations. Specifically, two-level temporal boundaries are provided to localize single action videos along with sub-steps of each action. More importantly, EgoExo-Fitness introduces innovative annotations for interpretable action judgement--including technical keypoint verification, natural language comments on action execution, and action quality scores. Combining all of these, EgoExo-Fitness provides new resources to study egocentric and exocentric full-body action understanding across dimensions of **what**, **when**, and **how well**. To facilitate research on egocentric and exocentric full-body action understanding, we construct benchmarks on a suite of tasks (\ie, action recognition, action localization, cross-view sequence verification, cross-view skill determination, and a newly proposed task of guidance-based execution verification), together with detailed analysis.

 ![](./img/dataset_intro.png)

## ⏬ Download
To download the dataset, please sign the [License Agreement](./License_Agreement.pdf) and send it to liym266@mail2.sysu.edu.cn for downloading our datasets and raw annotations. Click [here](./Raw_annotations/README.md) to learn more details about the released dataset and the raw annotations.


## 📑 Citation
Please cite it if you find this work useful.
```
@inproceedings{li2024egoexo,
  title={EgoExo-Fitness: towards egocentric and exocentric full-body action understanding},
  author={Li, Yuan-Ming and Huang, Wei-Jin and Wang, An-Lan and Zeng, Ling-An and Meng, Jing-Ke and Zheng, Wei-Shi},
  booktitle={European Conference on Computer Vision},
  pages={363--382},
  year={2024},
  organization={Springer}
}
```
