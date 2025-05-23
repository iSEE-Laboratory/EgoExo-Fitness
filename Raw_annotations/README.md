# Details about the released EgoExo-Fitness Dataset

## ⏬ Download
To download the dataset, please sign the [License Agreement](../License_Agreement.pdf) and send it to liym266@mail2.sysu.edu.cn for downloading our datasets and raw annotations. We provide the following downloading ways:
- **Baidu NetDisk**: After applying for the dataset, we will sent you a download link of Baidu NetDisk, which would be expired in one week.
- **Huggingface**: We have also uploaded our dataset on Huggingface. If you hope to download our dataset in this way, please sign the [License Agreement](../License_Agreement.pdf) together with your Huggingface UserName or Email, and then apply the dataset [here](https://huggingface.co/datasets/Lymann/EgoExo-Fitness).

## Data 
EgoExo-Fitness featrues synchronized egocentric and exocentric fitness videos. Through the provided link, you can download the following data:
- Preprocessed video frames in 30 fps.
- Extracted frame-wise CLIP-B features.

Currently the raw videos are not available through the link. If you are interested in the raw videos, please feel free to contact us.

## Statistics
We provide statistics calculating and drawing scripts in `./statistics_drawings.ipynb`.

## Raw Annotations
The raw annotations are also provided through the download link. Here are some illustrations of the raw annotations.

 ![](../img/statistics.png)

### Meta Records
`meta_records.json` includes basic information (e.g., record id, views, frames, etc) of all available records. Here is the example:

```
{
    "records": [
        {
            "record_id": "ThEnUZ",
            "views": [
                "ego_l",
                ...
            ],
            "frames": {
                "ego_l": {
                    "path": "frames_open/ThEnUZ/ego_l",
                    "num_frames": 7973
                },
                ...
            },
            "num_views": 6,
            "num_sequences": 3,
            "sequences": {
                "sequence_start_end_frame": [
                    [
                        20,
                        1000
                    ],
                    ...
                ]
            },
            "num_actions": 12
        },
        ...
  ],
  "record_index": {
        "ThEnUZ": 0,
        ...
  }
}
```

### Action-Level Boundaries
`action_level_annotations.json` includes action-level temporal boundaries. Here is the example:
```
{
    "CeqSkC": {         // The key is the record ID in meta_records.json
        "num_actions": 16,
        "action_info": [
            [
                1,      // action ID
                106,    // start frame ID
                496     // end frame ID
            ],
            ...
        ]
    },
    ...
}
```

### Substep-Level Boundaries
`substep_level_annotations.json` includes substep-level temporal boundaries. Note that we convert the annotations as [ActivityNet1.3](https://uwmadison.app.box.com/s/aisdoymowukc99zoc7gpqegxbb4whikx) format. Here is the example:
```
{
    "classes": [
        "Kneeling pushing-ups",
        "Push-ups",
        ...
    ],
    "database": {
        "FQPS6Y_4-3-1_ego_m": {
            "annotations": [
                {
                    "label": "Kneeling pushing-ups",
                    "segment_time": [
                        167,
                        182
                    ],
                    "segment_frame": [],
                    "segment": [
                        3.333333333333343,
                        18.333333333333343
                    ],
                    "fps": 30
                },
                ...
            ],
            "duration": 69.66666666666667,
            "fps": 30,
            "num_frames": 2090,
            "resolution": "",
            "subset": "test",
            "view": "ego_m",
            "actor": "FQPS6Y",
            "path": "frames_openFQPS6Y/ego_m/",
            "seq_st": 4910,
            "seq_ed": 7000
        },
        ...
    }
}
```

### Interpretable Action Judgement
`interpretable_action_judgement.json` includes detailed annotations on how well an single action is performed. Here is the example:
```
{
    "ThEnUZ_action_1": {
        "annotations": [
            {
                "key_point_verification": [
                    [
                        "Cross your feet.",
                        "True"
                    ],
                    [
                        "Keep your back straight.",
                        "False"
                    ],
                    ...
                ]
                "action_quality_score": 3,
                "comment": "The movement was performed according to the instructions, but the back was not kept straight enough and the depth of the descent was insufficient.",
                "action_name": "Kneeling pushing-ups",
                "action_guidance": "cross your feet, kneel on the mat, keep your back straight, keep your body in a straight line from the side, and put your hands on both sides of the chest, slightly wider than shoulder-width apart. bend your arms and bend down until your elbows are slightly above your torso, then stretch your arms and get up to restore. ",
                "annotator": "F03vpUuT3e"
            },
            ...
        ],
        "st_ed_frame": [
            241,
            691
        ],
        "frame_root": "frames_open/ThEnUZ"
    },
    ...
}
```

## 📑 Citation
Please cite it if you find this work useful.
```
@inproceedings{li2024egoexo,
  title={EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding},
  author={Li, Yuan-Ming and Huang, Wei-Jin and Wang, An-Lan and Zeng, Ling-An and Meng, Jing-Ke and Zheng, Wei-Shi},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
