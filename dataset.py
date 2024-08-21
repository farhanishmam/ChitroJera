def normalise_bn(text_bn):
    return normalize(
        text_bn,
        unicode_norm="NFKC",
        punct_replacement=None,
        url_replacement=None,
        emoji_replacement=None,
        apply_unicode_norm_last=True
    )
    
def load_dataset(path):
    train_df = pd.read_csv(path + "train.csv")
    val_df = pd.read_csv(path + "valid.csv")
    test_df = pd.read_csv(path + "test.csv")

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    all_labels = list(set(df['Answer_fixed' if CFG.lang == 'bn' else 'Answer_en'].unique().astype(str)))
    all_labels.sort()
    label_map = dict()
    CFG.num_classes = len(all_labels)
    for idx, label in enumerate(all_labels):
        label_map[normalise_bn(str(label)) if CFG.lang == 'bn' else str(label)] = idx
    
    return df, train_df, val_df, test_df
    
class VQADataset(Dataset):
    def __init__(self, features, img_dir, img_transform=None, caption_transform=None, target_transform=None):
        self.features = features
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.caption_transform = caption_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img_path = str(self.img_dir.joinpath(self.features['image_name'].iloc[idx]))
        image = read_image(img_path, mode=ImageReadMode.RGB).to(device=CFG.device)
        caption = normalise_bn(self.features['Question' if CFG.lang == 'bn' else 'Question_en'].iloc[idx])
        identity = self.features['image_name'].iloc[idx]
        label = torch.tensor(label_map[normalise_bn(str(self.features['Answer_fixed'].iloc[idx])) if CFG.lang == 'bn' else str(self.features['Answer_en'].iloc[idx])], dtype=torch.long)
        
        if self.img_transform:
            image = self.img_transform(image)
        if self.caption_transform:
            caption = self.caption_transform(caption)
        if self.target_transform:
            label = self.target_transform(label)
            
        processed_img = CFG.processor(images=image, return_tensors="pt")
        image = processed_img['pixel_values']
        
        processed_txt = CFG.tokenizer.encode_plus(
            caption,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        seq = processed_txt['input_ids']
        mask = processed_txt['attention_mask']
        
        return identity, image, seq, mask, label
    
  

class Collator(object):
    def __init__(self, test=False):
        self.test = test
    def __call__(self, batch):
        ids, images, seqs, masks, labels = zip(*batch)

        seqs = [seq.squeeze(dim=0) for seq in seqs]
        masks = [mask.squeeze(dim=0) for mask in masks]
        images = [image.squeeze(dim=0) for image in images]
        labels = torch.stack(labels)

        seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        masks = nn.utils.rnn.pad_sequence(masks, batch_first=True)

        images = torch.stack(images)
        
        return ids, images, seqs, masks, labels