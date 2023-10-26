1. Dataset:

    use font2img.py, generate_img.py to generate dataset

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

    /data/Font/byFont, /data/Font/byUnicode可直接使用，包含142 Fonts, 12948 words
    
2. Train:
    在train.py中指定gpu:
        " os.environ["CUDA_VISIBLE_DEVICES"] = "" "
    在config.yaml中指定參數：
        (1) train style or content classifier:
            n_classes of style classifier: 142
            n_classes of content classifier: 12948
        (2) use ddp: use ddp or not
        (3) others: batch_size, num_workers, warmup, accumulation_step...
    
    about model:
    train.py中，使用timm.create_model(), 詳細可使用模型可參考model.txt
    
    training detail of swin_transformer, convnext_large, efficientnetv2_m 可以參考result/from_handover/content, style裡的tensorboard record
    command:
    "tensorboard --logdir dir_name"

3. Test: