class CFG:
    save_models = True
    init_weights = True
    tokeniser_model_id = 'csebuetnlp/banglabert'
    text_model_id = 'csebuetnlp/banglabert'
    image_model_id = 'google/vit-base-patch16-224-in21k'
    tokeniser_model = None
    text_model = None
    image_model = None
    text_model_config = None
    image_model_config = None
    lang = 'bn'
    
    images_base_path = Path('/kaggle/input/vqa-bangla/Bangla_VQA/Bangla_VQA/images')
    images_base_path_test = Path('/kaggle/input/vqa-bangla/Bangla_VQA/Bangla_VQA/images')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    debug = False
    print_freq = 50
    apex = True # for faster training
    epochs = 15
    learning_rate = 2e-5  # for adam optimizer
    eps = 1e-6
    betas = (0.9, 0.999)  # for adam optimizer
    batch_size = 64
    max_len = 512
    weight_decay = 0.01  # for adam optimizer regulaization parameter
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    train = True
    num_classes = 0
    
    frozen_lm = False # True or False 
    fusion_mode = "co_attention" # "merged_attention", "co_attention"
    no_fusion_encoder = 2
    num_heads = 4
    
    mlp_hidden_size = 256
    mlp_hidden_layers = 0
    mlp_dropout = 0.1
    mlp_grad_clip = 1.0
    mlp_init_range = 0.2
    mlp_attn_dim = 256