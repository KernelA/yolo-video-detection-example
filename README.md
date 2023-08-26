# YOLO video detection in the TouchDesigner

## Description


## Requirements

1. Python 3.9 or higher. It is more preferable to have same version as in the TouchDesigner.

## How to run

### Install dependencies

For GPU:
```
pip install -r. /requirements.gpu.txt
```
s
For CPU:
```
pip install -r. /requirements.cpu.txt
```

For development:
```
pip install -r ./requirements.dev.txt
```

### Compile Cython extension

You need C compiler to compile extension. [See](https://docs.cython.org/en/latest/src/quickstart/install.html#installing-cython)

```
python -m cibuildwheel --platform <your_platform>
```

Install binary distribution:
```
pip install ./dist/*.whl
```

## Usecases

### Video detection
Run:
```
python ./main.py -c <checkpoint_path> -i <video_path> -o <video_path>
```

**If you run script in the directory with source code see modification of `sys.path` in the beginner of the file.**

### With TouchDesigner

Run server processing:
```
python ./processing.py -p <path_to_model>
```

Open `touch_designer.py` in the TouchDesigner as SCript TOP.