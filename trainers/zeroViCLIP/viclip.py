import os
import logging
import torch
from einops import rearrange
from torch import nn
import math

# from .criterions import VTC_VTM_Loss
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from viclip_vision import clip_joint_l14
from viclip_text import clip_text_l14
import torch
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class ViCLIP(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self,  
                 tokenizer=None, 
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/mnt/new_nas/yansan/models/ViCLIP/ViCLIP-L_InternVid-FLT-10M.pth"),
                 freeze_text=True):
        super(ViCLIP, self).__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = _Tokenizer()
        self.max_txt_l = 32

        if size.lower() == 'l':
            self.vision_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.vision_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")
    
        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.text_width = 768 
        self.embed_dim = 768 
        self.masking_prob = 0.9
        
        if size.lower() == 'l':
            self.text_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.text_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")
        
        self.text_encoder_pretrained = False#'bert-base-uncased'
        self.text_encoder_d_model = 768

        self.text_encoder_vocab_size = 49408
        
        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()

        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0

        if pretrain:
            logger.info(f"Load pretrained weights from {pretrain}")
            state_dict = torch.load(pretrain, map_location='cpu')['model']
            self.load_state_dict(state_dict)
        
        # Freeze weights
        if freeze_text:
            self.freeze_text()


    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )

        return ret

    def forward(self, image, text, raw_text, idx, log_generation=None, return_sims=False):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()

        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(raw_text)
        if return_sims:
            sims = torch.nn.functional.normalize(vision_embeds, dim=-1) @ \
                  torch.nn.functional.normalize(text_embeds, dim=-1).transpose(0, 1)
            return sims

        # calculate loss

        ## VTC loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        device = next(self.text_encoder.parameters()).device
        text = self.text_encoder.tokenize(
            text, context_length=self.max_txt_l
        ).to(device)
        text_embeds = self.text_encoder(text)
        return text_embeds

    @torch.no_grad()
    def clip_contrastive_temperature(self, min_val=0.001, max_val=0.5):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")
            
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        
        if encoder_name == "vit_l14":
            text_encoder = clip_text_l14(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
            )
        elif encoder_name == "vit_b16":
            text_encoder = clip_text_b16(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_text_features(self, input_text, tokenizer):
        text_template= f"{input_text}"
        with torch.no_grad():
            text_features = self.encode_text(text_template).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)      
        return text_features

    # def get_text_features(self, input_text, tokenizer, text_feature_dict={}):
    #     if input_text in text_feature_dict:
    #         return text_feature_dict[input_text]
    #     text_template= f"{input_text}"
    #     with torch.no_grad():
    #         # text_token = tokenizer.encode(text_template).cuda()
    #         text_features = self.encode_text(text_template).float()
    #         text_features /= text_features.norm(dim=-1, keepdim=True)      
    #         text_feature_dict[input_text] = text_features
    #     return text_features

    def get_vid_features(self, input_frames):
        with torch.no_grad():
            clip_feat = self.encode_vision(input_frames,test=True).float()
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
        return clip_feat

    def get_predict_label(self, clip_feature, text_feats_tensor, top=5):
        label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
        return top_probs, top_labels

    
if __name__ =="__main__":
    def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
        for t in texts:
            feat = clip.get_text_features(t, tokenizer, text_feat_d)
            text_feat_d[t] = feat
        return text_feat_d

    def get_vid_feat(frames, clip):
        return clip.get_vid_features(frames)

    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    def normalize(data):
        return (data/255.0-v_mean)/v_std

    def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
        assert(len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
        return vid_tube

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _Tokenizer()
    vclip = ViCLIP(tokenizer=tokenizer, pretrain="/mnt/new_nas/yansan/models/ViCLIP/ViClip-InternVid-10M-FLT.pth", freeze_text=True) 
    vclip = vclip.to(device)
    video = cv2.VideoCapture('uZA6TYgnsGUAAAAAAAAAAAAAVoeUAQBr.mp4')
    frames = [x for x in _frame_from_video(video)]
    text_candidates = ["mooncake",
                "pastries",
                "products",
                "beef",
                "food",
                "household electrical appliances",
                "grain and oil seasoning/fast food/dry goods/baking",
                   ]


    frames_tensor = frames2tensor(frames, device=device)

    print(frames_tensor.shape)
    vid_feat = vclip.get_vid_features(frames_tensor)

    for t in text_candidates:
        txt_feat = vclip.get_text_features(t, tokenizer)
        similarity = vid_feat @ txt_feat.t()
        print(t, similarity.item())