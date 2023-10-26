content/style classifier Directions
1. torch version:
    for this sample code, use torch==1.13.1+cu116
2. assign gpu:
    e.g. os.environ["CUDA_VISIBLE_DEVICES"] = "NUM", NUM is gpu id
3. choose style/content mode with parser 'mode'
    'content' to use content classifer, 'style' to use ctyle classifier
4. modify data_path for style/content classifer
5. there is a example dataset, use 'tar zxvf data.tar.gz' to unzip

Example directory hierarchy：
    (1) style classifier
    data_dir
        |--- font1
        |--- font2
            |--- 4e00.png
            |--- 4e01.png
            |--- ...
        |--- ...
    (2) content classifier
    data_dir
        |--- 4e00
        |--- 4e01
            |--- font1.png
            |--- font2.png
            |--- ...
        |--- ...

for testing dataset:
content classifier acc: 99.535%
style classifier acc: 99.942%