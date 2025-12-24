from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']


class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx, video_paths=None):
        
        
        self.label_ids = label_ids
        self.text_feats = text_feats
        self.cons_text_feats = cons_text_feats
        self.condition_idx = condition_idx
        self.video_feats = video_feats
        self.audio_feats = audio_feats
        self.size = len(self.text_feats)
        if video_paths is not None:
            self.video_paths = video_paths
        else:
            self.video_paths = [None] * self.size  # 确保长度一致

            # 验证长度一致性
        if len(self.video_paths) != self.size:
            raise ValueError(f"video_paths长度({len(self.video_paths)})与样本数({self.size})不匹配")
        # self.video_paths = video_paths if video_paths else []  # 列表 of str

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats['feats'][index]),
            'audio_feats': torch.tensor(self.audio_feats['feats'][index]),
            'cons_text_feats': torch.tensor(self.cons_text_feats[index]),
            'condition_idx': torch.tensor(self.condition_idx[index]),
            'video_paths': self.video_paths[index] if self.video_paths else None,  # 新增单个路径str
        } 

        return sample


