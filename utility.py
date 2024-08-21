from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
 
OUTPUT_DIR = "./"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def resize_images(img_tensor):
    return resizer(img_tensor)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
    
def get_score(y_trues, y_preds):
    accuracy = accuracy_score(y_trues, y_preds)
    return accuracy
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger 
        


def bert_scorer(df):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_list = []
    
    for index, row in df.iterrows():
        original = CFG.tokenizer.encode_plus(normalise_bn(str(row['label'])), return_tensors="pt").to(CFG.device)
        preds = CFG.tokenizer.encode_plus(normalise_bn(str(row['pred'])), return_tensors="pt").to(CFG.device)
        
        with torch.no_grad():
            d1 = CFG.text_model_vanilla(original['input_ids'], attention_mask=original['attention_mask'], output_hidden_states=True).hidden_states[-1][:, 0, :]
            d2 = CFG.text_model_vanilla(preds['input_ids'], attention_mask=preds['attention_mask'], output_hidden_states=True).hidden_states[-1][:, 0, :]
        
        sim_list.append(cos(d1, d2).item())
        
    return sim_list
    
def save_predictions(model_name, ids, labels, preds, split='val'):
    entries = []
    for identity, label, pred in zip(ids, labels, preds):
        entry = {
            'identity': identity,
            'label': all_labels[label],
            'pred': all_labels[pred]
        }
        entries.append(entry)

    with open(f'/kaggle/working/{model_name}_{split}_preds.json', 'w') as fp:
        json.dump(entries, fp, cls=NpEncoder)
        
    preds_df = pd.DataFrame.from_dict(entries)
    similarity_list = bert_scorer(preds_df)
    
    report_dict = classification_report(labels, preds, digits=4, zero_division=0, output_dict=True)
    report_dict['bert_score'] = {
        'model': CFG.text_model_config._name_or_path,
        'bert_score_mean': sum(similarity_list) / len(similarity_list)
    }
    
    with open(f'/kaggle/working/{model_name}_{split}_results.json', 'w') as fp:
        json.dump(report_dict, fp)