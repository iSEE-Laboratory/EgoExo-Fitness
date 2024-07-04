# Raw annotations of EgoExo-Fitness Dataset

## Meta Records
`meta_records.json` include basic information (e.g., record id, views, frames, etc) of all available records. Here is the example:

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
