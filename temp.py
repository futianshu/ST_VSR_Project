    model = ST_VSR_Network().to(device)
    load_dpas_sr_prior(model, "/home/ubuntu/lib/hsh/TSD-SR/checkpoint/tsdsr/vae.safetensors")
    
    # ==============================================================
    # 🚀 1. 数据集初始化与 Loss 实例化 (保持不变)
    # ==============================================================
    vimeo_root = "/home/ubuntu/data/OpenDataLab___Vimeo90K/raw/vimeo_septuplet"
    train_dataset = Vimeo90K_ST_Dataset(data_root=vimeo_root, scale=4, patch_size=128)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_dataset = Vimeo90K_ST_Val_Dataset(data_root=vimeo_root, scale=4)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    for param in loss_fn_vgg.parameters(): param.requires_grad = False
    charbonnier = CharbonnierLoss().to(device) 
    
    # ==============================================================
    # 🚀 2. 定义优化器
    # ==============================================================
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # ==============================================================
    # 🚀 3. 加载断点权重 (重点：一定要在 Compile 之前加载基础模型权重！)
    # ==============================================================
    resume_epoch = 0 
    start_epoch = 1
    best_psnr = 0.0  
    
    ema_state_dict_cache = None # 缓存 EMA 权重
    
    if resume_epoch > 0:
        checkpoint_path = f"checkpoints/st_vsr_epoch_{resume_epoch}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # 先加载基础模型权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'best_psnr' in checkpoint: best_psnr = checkpoint['best_psnr']
            
            # 将 EMA 权重取出来暂存，等一会儿实例化后再加载
            if 'ema_model_state_dict' in checkpoint:
                ema_state_dict_cache = checkpoint['ema_model_state_dict']
                
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 成功加载完整断点：{checkpoint_path}，当前最佳 PSNR: {best_psnr:.2f}")

    # ==============================================================
    # 🚀 4. 编译优化与 EMA 实例化
    # ==============================================================
    try:
        model = torch.compile(model)
        print("✅ 模型已启用 torch.compile 编译优化！")
    except Exception as e:
        print(f"⚠️ torch.compile 启用失败: {e}")
        
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)) 
    
    # 现在将缓存的 EMA 权重完美加载进去
    if ema_state_dict_cache is not None:
        ema_model.load_state_dict(ema_state_dict_cache, strict=False)
        print("✅ 成功恢复 EMA 平滑权重历史！")

    print("🔥 真实世界时空联合训练与验证正式开始！")