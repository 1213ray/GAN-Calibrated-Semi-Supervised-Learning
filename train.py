if __name__ == '__main__':
    import ultralytics
    from ultralytics import YOLO

    dataloader_kwargs = dict(
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = YOLO('models/yolov8s.pt')
    results = model.train(
        data="yaml/person.yaml",
        imgsz=640,
        epochs=400,
        patience=20,
        batch=16,
        workers=4,  # 啟用 12 個子進程來加載數據，您可以根據 CPU 核心數調整
        half=True, # 啟用混合精度訓練
        project='result',
        name='person',

    )
