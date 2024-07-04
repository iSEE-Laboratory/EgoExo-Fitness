# Raw annotations of EgoExo-Fitness Dataset

## Meta Records
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
            "num_views": 6
            "num_sequences": 3
        },
        ...
  ],
  "record_index": {
        "ThEnUZ": 0,
        ...
  }
}
```

## Interpretable Action Judgement
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
