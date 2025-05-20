def get_model_from_config(cfg, num_classes):
    """
    Dynamically initialize detection model from config.
    Supports CMNeXtFasterRCNN, CMNeXtDETR, etc.
    """
    model_name = cfg['MODEL']['NAME'].lower()
    model_kwargs = cfg['MODEL']
    modals = cfg['DATASET']['MODALS']
    criterion_cfg = cfg.get('CRITERION', {})

    if model_name == 'cmnextfasterrcnn':
        from semseg.models.cmnext_detection import CMNeXtFasterRCNN
        return CMNeXtFasterRCNN(
            backbone_name=model_kwargs['BACKBONE'],
            num_classes=num_classes,
            modals=modals
        )

    elif model_name == 'cmnextdetr':
        from semseg.models.cmnext_detection import CMNeXtDETR
        return CMNeXtDETR(
            backbone=model_kwargs['BACKBONE'],
            num_classes=num_classes,
            num_queries=model_kwargs.get('NUM_QUERIES', 300),
            hidden_dim=model_kwargs.get('HIDDEN_DIM', 256),
            num_feature_levels=model_kwargs.get('NUM_FEATURE_LEVELS', 4),
            criterion_cfg=criterion_cfg,
            modals=modals
        )
    
    elif model_name == 'cmnextretinanet':
        from semseg.models.cmnext_detection import CMNeXtRetinaNet  
        from semseg.models.cmnext_detection import CMNeXtBackbone

        backbone = CMNeXtBackbone(
            backbone=model_kwargs['BACKBONE'],
            modals=modals
        )
        return CMNeXtRetinaNet(
            backbone=backbone,
            num_classes=num_classes
        ) 

    else:
        raise ValueError(f"Unsupported model name: {cfg['MODEL']['NAME']}")
