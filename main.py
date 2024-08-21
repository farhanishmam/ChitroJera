def main():
    torch.autograd.set_detect_anomaly(True)
    #transformers.__version__

    seed_everything(CFG.seed)
    # Initialize the image resizer
    resizer = Resize((224, 224), antialias=True)
    
    path = "/path_to_data"
    df, train_df, val_df, test_df = load_dataset(path)
    
    # Instantiate datasets
    train_dataset = VQADataset(train_df, CFG.images_base_path, img_transform=resizer)
    val_dataset = VQADataset(val_df, CFG.images_base_path, img_transform=resizer)
    test_dataset = VQADataset(test_df, CFG.images_base_path, img_transform=resizer)

    LOGGER = get_logger()
    
    collate = Collator()
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate)
    valid_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)

    loss_fn = nn.CrossEntropyLoss()
    
    model_id = 'co_attention__itm+mlm+ucl_loss1.0739_best'
    backbone_model = BanCAP_Pretraining(CFG).to(CFG.device)
    backbone_model = nn.DataParallel(backbone_model)
    backbone_model.load_state_dict(torch.load(f'/kaggle/input/bancap-pretraining-for-banvqa/{model_id}.pth', map_location=torch.device(CFG.device))['model'])

    model = BanCAP_Pretraining_Classifier(backbone_model, CFG).to(CFG.device)
    optim = AdamW(model.parameters(), lr=CFG.learning_rate, eps=CFG.eps, betas=CFG.betas)
    
    train(model, optim, loss_fn, train_dataloader, epoch)

if __name__ == "__main__":
    main()