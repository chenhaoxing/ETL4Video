from MMA_viclip import MMA_ViCLIP 
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
import json
from einops import rearrange
import os
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, viclip_model):
        super().__init__()
        self.transformer = viclip_model.text_encoder.transformer
        self.positional_embedding = viclip_model.text_encoder.positional_embedding
        self.ln_final = viclip_model.text_encoder.ln_final
        self.text_projection = viclip_model.text_encoder.text_projection

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, viclip_model):
        super().__init__()
        self.n_cls = len(classnames)
        clip_imsize = viclip_model.inputs_image_res
        self._build_text_embedding(cfg, classnames, viclip_model)
        self.description_s = cfg.DATA.DES_SPATIAL
        self.description_t = cfg.DATA.DES_TEMPORAL

        self.N = cfg.TRAINER.MMADAPTER.DES_N

        self._build_spatil_text_embeddings(classnames, viclip_model)
        self._build_temporal_text_embeddings(classnames, viclip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            viclip_model.text_encoder.ln_final.weight.shape[0], 
            len(viclip_model.text_encoder.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_mm_adapter(
            viclip_model.vision_encoder.ln_post.weight.shape[0],
            len(viclip_model.vision_encoder.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM
        )

        self.shared_adapter = self._build_adapter(
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            len(viclip_model.vision_encoder.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM
        )
        
        self.adapter_scale = float(cfg.TRAINER.MMADAPTER.ADAPTER_SCALE)

    def _build_spatil_text_embeddings(self, classnames,viclip_model):
        # text_aug = 'This is a video about {}'

        text_aug = 'A video of {}.'
        classes = []
        with open(self.description_s, 'r') as f:
            description_spatial = json.load(f)
        f.close()

        for key, category_name in zip(description_spatial, classnames):
            buffer = []
            for capt in description_spatial[key]:
                buffer.append(viclip_model.text_encoder.tokenize(capt, context_length=32))
                # buffer.append(viclip_model.text_encoder.tokenize(text_aug.format(category_name[1]) + capt, context_length=32))
            classes.append(torch.stack(buffer).squeeze())
        classes = torch.stack(classes)
        # b,N,d -> bN, d
        classes_token = rearrange(classes, 'a n d -> (a n) d')

        with torch.no_grad():
            # bN, d, m
            embedding = viclip_model.text_encoder.token_embedding(classes_token.cuda())

        self.register_buffer("spatil_token_embedding", embedding)
        self.register_buffer("spatil_tokenized_prompts", classes_token)

    def _build_temporal_text_embeddings(self, classnames, viclip_model):
        text_aug = 'A video of {} usually includes {}'
        classes = []
        with open(self.description_t, 'r') as f:
            description_temporal = json.load(f)
        f.close()

        for key, category_name in zip(description_temporal, classnames):
            buffer = []
            for capt in description_temporal[key]:
                # buffer.append(viclip_model.text_encoder.tokenize(text_aug.format(category_name[1], capt), context_length=32))
                buffer.append(viclip_model.text_encoder.tokenize(capt, context_length=32))
            classes.append(torch.stack(buffer).squeeze()) 
        classes = torch.stack(classes)

        classes_token = rearrange(classes, 'a n d -> (a n) d')

        with torch.no_grad():
            embedding = viclip_model.text_encoder.token_embedding(classes_token.cuda())

        self.register_buffer("temporal_token_embedding", embedding)
        self.register_buffer("temporal_tokenized_prompts", classes_token)

    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale
        # return self.text_adapter[index], None, self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale
        # return self.visual_adapter[index], None, self.adapter_scale

    def _build_text_embedding(self, cfg, classnames, viclip_model):
        # text_aug = f"{{}}"

        text_aug = 'a video of a person doing {}.'
        tokenized_prompts = torch.cat([viclip_model.text_encoder.tokenize(text_aug.format(c), context_length=32) for c in classnames])
        with torch.no_grad():
            embedding = viclip_model.text_encoder.token_embedding(tokenized_prompts.cuda())
        
        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dropout_rate=0.1):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(
                        nn.Linear(d_model, mid_dim),
                        nn.ReLU()
                    )),
                    ("dropout", nn.Dropout(p=dropout_rate)),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        return adapter

    def _build_mm_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dropout_rate=0.1):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(
                        nn.Linear(d_model, mid_dim),
                        nn.ReLU()
                    )),
                    ("dropout", nn.Dropout(p=dropout_rate)),
                    ("spatio_up", nn.Linear(mid_dim, d_model)),
                    ("temporal_up", nn.Conv3d(mid_dim, d_model, kernel_size=(3,1,1), stride=(1, 1, 1), padding='same', groups=mid_dim, dilation=(1, 1, 1)))
                ]))

        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        return adapter

    def forward(self):
        spatil_embedding = self.spatil_token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(spatil_embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            spatil_embedding = spatil_embedding + self.adapter_scale * token_embedding
    
        temporal_embedding = self.temporal_token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(temporal_embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            temporal_embedding = temporal_embedding + self.adapter_scale * token_embedding
    
        embedding = self.token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding
    
        return embedding, spatil_embedding, temporal_embedding, self.text_adapter_func, self.visual_adapter_func



class CustomViCLIP(nn.Module):
    def __init__(self, cfg, classnames, viclip_model):
        super().__init__()
        self.cfg = cfg 
        self.adapter_learner = AdapterLearner(cfg, classnames, viclip_model)

        self.spatil_tokenized_prompts = self.adapter_learner.spatil_tokenized_prompts
        self.temporal_tokenized_prompts = self.adapter_learner.temporal_tokenized_prompts

        self.tokenized_prompts = self.adapter_learner.tokenized_prompts

        self.image_encoder = viclip_model.vision_encoder
        self.text_encoder = TextEncoder(viclip_model)
        self.N = cfg.TRAINER.MMADAPTER.DES_N
        self.logit_scale = 100

    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features

    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                image, visual_adapter_func, masking_prob=0.0
            )
        else:
            image_features = self.image_encoder(
                image
            )
        return image_features

    def inference(self, image):
        token_embedding, spatil_token_embedding, temporal_token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
        temporal_tokenized_prompts, spatil_tokenized_prompts, tokenized_prompts = self.temporal_tokenized_prompts, self.spatil_tokenized_prompts, self.tokenized_prompts

        image = image.permute(0, 2, 1, 3, 4).contiguous()

        text_features = self.encode_text(
                token_embedding, tokenized_prompts, text_adapter_func
            )

        image_features = self.encode_image(image, visual_adapter_func)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, image):

        token_embedding, spatil_token_embedding, temporal_token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
        temporal_tokenized_prompts, spatil_tokenized_prompts, tokenized_prompts = self.temporal_tokenized_prompts, self.spatil_tokenized_prompts, self.tokenized_prompts

        b, t, c, h, w = image.size()
        image = image.permute(0, 2, 1, 3, 4).contiguous()

        text_features = self.encode_text(
                token_embedding, tokenized_prompts, text_adapter_func
            )

        spatil_text_features = self.encode_text(
                spatil_token_embedding, spatil_tokenized_prompts
            )

        temporal_text_features = self.encode_text(
                temporal_token_embedding, temporal_tokenized_prompts
            )

        spatil_text_features = rearrange(spatil_text_features, '(a n) d -> a n d', n=self.N)
        temporal_text_features = rearrange(temporal_text_features, '(a n) d -> a n d', n=self.N)
        spatil_text_features = spatil_text_features.mean(dim=1, keepdim=False)
        temporal_text_features = temporal_text_features.mean(dim=1, keepdim=False)

        image_features = self.encode_image(image, visual_adapter_func)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        spatil_text_features = spatil_text_features / spatil_text_features.norm(dim=-1, keepdim=True)
        temporal_text_features = temporal_text_features / temporal_text_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale * image_features @ text_features.t()

        if self.cfg.TRAIN.Cons_LOSS:
            return logits, spatil_text_features, temporal_text_features, text_features

        return logits

def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6

def return_MMA_ViCLIP(config, logger=None, class_names=None):
    logger.info(f"Loading ViCLIP")
    viclip_model = MMA_ViCLIP(pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/mnt/new_nas/yansan/models/ViCLIP/ViCLIP-B_InternVid-FLT-10M.pth"))
    viclip_model = viclip_model.cuda()
    logger.info("Building Adapter Tuning ViCLIP")
    model = CustomViCLIP(config, class_names, viclip_model)

    # 冻结除 adapter 之外的参数
    for name, param in model.named_parameters():
        if "text_adapter" not in name and "visual_adapter" not in name and "shared_adapter" not in name:
            param.requires_grad_(False)

    trainable_params = count_parameters_in_millions(model)
    logger.info(f"Total learnable parameters: {trainable_params} MB")

    return model


