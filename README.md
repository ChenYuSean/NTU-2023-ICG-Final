# NTU 2023 Spring ICG Final

## Execute
```bash
usage: main.py [-h] [--output OUTPUT] [--alpha ALPHA] [--frame FRAME] [--morpher {0,1}] [--debug]
               source target

Morphing

positional arguments:
  source           Path to the source image
  target           Path to the target image

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Path to the output directory
  --alpha ALPHA    Alpha value for morphing
  --frame FRAME    Number of frames for morphing, overwrites alpha value if set
  --morpher {0,1}  Morphing method
  --debug          Run debug mode
```


Examples:
```bash
python main.py ./data/ffhq_00064.png ./data/ffhq_00114.png --alpha 0.5
```

## Dataset
https://www.kaggle.com/datasets/atulanandjha/lfwpeople
https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq