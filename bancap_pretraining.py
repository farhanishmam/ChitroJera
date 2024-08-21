class BanCAP_Pretraining(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.itm = ITMHead(self.cfg.text_model_config.hidden_size)
        self.text_model = self.cfg.text_model
        self.image_model = self.cfg.image_model
        
        self.hidden_size = 768
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.l_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        if self.cfg.fusion_mode == "merged_attention":
            self.fusion_encoder = nn.TransformerEncoderLayer(d_model=self.hidden_size, 
                                                        nhead=self.cfg.num_heads)
            self.fusion_layers = nn.TransformerEncoder(self.fusion_encoder, 
                                                  num_layers=self.cfg.no_fusion_encoder)
        
        elif self.cfg.fusion_mode == "co_attention":
            self.pool = MeanPooling()
            self.visual_cross_encoder = CrossAttentionEncoder(self.hidden_size, self.hidden_size,
                                                        self.cfg.no_fusion_encoder,
                                                        self.cfg.num_heads)
            
            self.language_cross_encoder = CrossAttentionEncoder(self.hidden_size, self.hidden_size,
                                                        self.cfg.no_fusion_encoder,
                                                        self.cfg.num_heads)
        
        
        self.text_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.vision_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        
        if self.cfg.frozen_lm:
            self._frozen_lm()
            
    def _frozen_lm(self):
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        for param in self.image_model.parameters():
            param.requires_grad = False
            

    def forward(self, input_ids, attention_mask, image):
        
        text_output = self.text_model(input_ids=input_ids, 
                                      attention_mask=attention_mask,
                                      output_hidden_states=True)
        image_output = self.image_model(image)

        visual_features = image_output.last_hidden_state
        text_features = text_output.hidden_states[-1]
        
        visual_embeds = visual_features[:, 0, :]
        text_embeds = text_features[:, 0, :]
        
        visual_poj = self.v_proj(visual_features)
        language_proj = self.l_proj(text_features)
  
        if self.cfg.fusion_mode == "co_attention":
            vision_final = self.visual_cross_encoder(visual_poj, language_proj)
            language_final = self.language_cross_encoder(language_proj, visual_poj)
            
            vision_final = torch.mean(vision_final, dim = 1)
            language_final = self.pool(language_final, attention_mask)
            
            vision_final = self.vision_mlp(vision_final)
            language_final = self.text_mlp(language_final)
            
        elif self.cfg.fusion_mode == "merged_attention":
            merged_embed = torch.cat((language_proj, visual_poj), dim = 1)
            merged_attention_features = self.fusion_layers(merged_embed)
            
            text_embed_len = attention_mask.size(1)
            language_final = merged_attention_features[:, :text_embed_len, : ]
            vision_final = merged_attention_features[:, text_embed_len:, : ]
            
            vision_final = self.vision_mlp(vision_final)
            language_final = self.text_mlp(language_final)
        
        return language_final, vision_final
        
class BanCAP_Pretraining_Classifier(torch.nn.Module):
    def __init__(self, backbone_model, cfg):
        super().__init__()
        self.backbone_model = backbone_model
        self.hidden_size = 768
        self.cfg = cfg
        self.classification_head = nn.Linear(2 * self.hidden_size, self.cfg.num_classes)
        
    def forward(self, input_ids, attention_mask, image):
        language_final, vision_final = self.backbone_model(input_ids, attention_mask, image)
        mm_final = torch.cat((language_final, vision_final), dim=1)
        output = self.classification_head(mm_final)
        
        return output