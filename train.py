def train_loop(model, optimizer, loss_fn, train_dataloader, epoch):
    model.train()
    train_losses = AverageMeter()
    start = end = time.time()
    
    for step, (_, image, seq, mask, label) in enumerate(tqdm(train_dataloader)):
        train_image = image.to(CFG.device)
        train_seq = seq.to(CFG.device)
        train_mask = mask.to(CFG.device)
        train_label = label.to(device=CFG.device)
        
        batch_size = train_image.shape[0]

        with torch.cuda.amp.autocast(enabled=CFG.apex):
            output = model(train_seq, train_mask, train_image)
            
        loss = loss_fn(output, train_label)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CFG.mlp_grad_clip)
        optimizer.step()
        
        train_losses.update(loss.item(), batch_size)
            
        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(train_dataloader) - 1):
            print(f'Epoch: [{epoch + 1}][{step}/{len(train_dataloader)}] '
                  f'Elapsed {timeSince(start, float(step + 1) / len(train_dataloader)):s} '
                  f'Loss: {train_losses.val:.4f} ({train_losses.avg:.4f}) ')
        
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return train_losses.avg
    
def validation_loop(model, loss_fn, valid_dataloader, epoch):
    all_ids = []
    all_preds = []
    all_labels = []
    
    model.eval()
    validation_losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    for step, (identity, image, seq, mask, label) in enumerate(tqdm(valid_dataloader)):
        image = image.to(device=CFG.device)
        seq = seq.to(device=CFG.device)
        mask = mask.to(device=CFG.device)
        label = label.to(device=CFG.device)
        
        batch_size = image.shape[0]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                output = model(seq, mask, image)

        loss = loss_fn(output, label)
        
        validation_losses.update(loss.item(), batch_size)
        predicted = output.argmax(dim=1)

        all_ids += list(identity)
        all_labels.append(label)
        all_preds.append(predicted)
            
        end = time.time()

        if step % CFG.print_freq == 0 or step == (len(valid_dataloader) - 1):
            print(f'Epoch: [{epoch + 1}][{step}/{len(valid_dataloader)}] '
                  f'Elapsed {timeSince(start, float(step + 1) / len(valid_dataloader)):s} '
                  f'Loss: {validation_losses.val:.4f} ({validation_losses.avg:.4f})')
        
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    all_preds_np = all_preds.cpu().numpy().astype(int)
    all_labels_np = all_labels.cpu().numpy().astype(int)
        
    return validation_losses.avg, all_ids, all_preds_np, all_labels_np
    
def train(model, optim, loss_fn, train_dataloader, epoch):
    best_score = float('-inf')

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train function 
        avg_train_loss = train_loop(model, optim, loss_fn, train_dataloader, epoch)

        # val function 
        avg_val_loss, all_ids, all_preds_np, all_labels_np = validation_loop(model, loss_fn, valid_dataloader, epoch)
        
        score = get_score(all_labels_np, all_preds_np)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

        if CFG.save_models and score > best_score:
            model_name = model_id + f'_score_{score:.4f}' + f'_{CFG.lang}'
            torch.save({'model': model.state_dict()}, f'{model_name}.pth')
            print(f'Saved model: {model_name}')
            best_score = score
            
            save_predictions(model_name, all_ids, all_labels_np, all_preds_np)
            
            avg_test_loss, all_ids_test, all_preds_np_test, all_labels_np_test = validation_loop(model, loss_fn, test_dataloader, epoch)
            save_predictions(model_name, all_ids_test, all_labels_np_test, all_preds_np_test, split='test')

        torch.cuda.empty_cache()
        gc.collect()