## **WRA-Net: Wide Receptive Field Attention Network for Motion Deblurring in Crops and Weeds Images**

official pytorch implement of WRA-Net

paper[link]

[model img]

---

## training

```python
python3 trainer.py --config [config path]
```

## test

```python
python3 test.py --data_dir [data directory path] --save_dir [path to save result]
```

---

## data

The data directory should be structured as follows.

```
-- CWFID
    |-- test
    |   |-- input
    |   |   |-- 001_image.png
    |   |   |-- 003_image.png
    |   |   |     ...
    |   |   |-- 035_image.png
    |   |   `-- 039_image.png 
    |   |-- label
    |   |   |-- 001_image.png
    |   |   |-- 003_image.png
    |   |   |--    ...
    |   |   |-- 035_image.png
    |   |   `-- 039_image.png
    |   |-- target
    |       |-- 001_image.png
    |       |-- 003_image.png
    |       |      ...
    |       |-- 035_image.png
    |       `-- 039_image.png
    |   
    |-- train
    |   |-- input
    |   |   |-- 002_image.png
    |   |   |-- 002_image_2.png
    |   |   |     ...
    |   |   |-- 060_image_4.png
    |   |   `-- 060_image_5.png
    |   |-- label
    |   |   |-- 002_image.png
    |   |   |-- 002_image_2.png
    |   |   |     ... 
    |   |   |-- 060_image_4.png
    |   |   `-- 060_image_5.png
    |   `-- target
    |       |-- 002_image.png
    |       |-- 002_image_2.png
    |       |     ..
    |       |-- 060_image_4.png
    |       `-- 060_image_5.png
    `-- val
        |-- input
        |   |-- 010_image.png
        |   |     ...
        |   `-- 030_image.png
        |-- label
        |   |-- 010_image.png
        |   |     ...
        |   `-- 030_image.png
        `-- target
            |-- 010_image.png
            |     ...
            `-- 030_image.png
```