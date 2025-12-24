import os

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor
import warnings
import cv2
from PIL import Image
import numpy as np
import logging

warnings.filterwarnings('ignore')


class MultimodalLLM_Enhancer(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_dim = args.text_feat_dim
        self.use_audio = getattr(args, 'use_audio_mllm', False)
        self.logger = logging.getLogger(args.logger_name)

        # 获取模型路径配置
        self.mllm_model_name = getattr(args, 'mllm_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        self.mllm_local_path = getattr(args, 'mllm_local_path', None)

        print(f"初始化多模态增强器，使用模型: {self.mllm_model_name}")
        print(f"本地模型路径: {self.mllm_local_path if self.mllm_local_path else '未指定，将从HuggingFace下载'}")

        try:
            # 优先从本地加载，如果本地路径存在
            if self.mllm_local_path and os.path.exists(self.mllm_local_path):
                print(f"从本地路径加载模型: {self.mllm_local_path}")
                self.processor = AutoProcessor.from_pretrained(
                    self.mllm_local_path,
                    trust_remote_code=True
                )
                self.visual_llm = AutoModelForVision2Seq.from_pretrained(
                    self.mllm_local_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # 否则从HuggingFace加载
                print(f"从HuggingFace加载模型: {self.mllm_model_name}")
                self.processor = AutoProcessor.from_pretrained(
                    self.mllm_model_name,
                    trust_remote_code=True
                )
                self.visual_llm = AutoModelForVision2Seq.from_pretrained(
                    self.mllm_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            # 冻结模型参数
            for param in self.visual_llm.parameters():
                param.requires_grad = False
            self.visual_llm.eval()

            # 记录模型信息
            total_params = sum(p.numel() for p in self.visual_llm.parameters())
            trainable_params = sum(p.numel() for p in self.visual_llm.parameters() if p.requires_grad)
            print(f"✓ MLLM模型加载成功")
            print(f"  总参数: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,} (冻结状态)")

        except Exception as e:
            print(f"✗ MLLM模型加载失败: {e}")
            print("  将回退到原始特征，不使用MLLM增强")
            self.visual_llm = None
            self.processor = None

        if self.use_audio:
            try:
                self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.audio_llm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.audio_llm.to(self.device)
                self.audio_llm.eval()
                self.logger.info("Whisper 加载成功")
            except Exception as e:
                self.logger.error(f"Whisper 加载失败: {e}")
                self.audio_llm = None

        self.video_semantic_proj = nn.Sequential(
            nn.Linear(4096, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(0.1)
        )

        if self.use_audio:
            self.audio_semantic_proj = nn.Sequential(
                nn.Linear(1024, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(0.1)
            )

        self.enhancer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
        )

        self.logger.info("多模态增强模块初始化完成")

    def load_video_frames(self, video_path, num_frames=3):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.warning(f"无法打开视频: {video_path}")
            return None
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // num_frames)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            if len(frames) >= num_frames:
                break
        cap.release()
        return frames if frames else None

    def extract_video_semantics(self, video_paths, batch_size=4):
        if self.visual_llm is None:
            return None
        video_semantics = []
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            batch_frames = [self.load_video_frames(path) for path in batch_paths]
            batch_frames = [f for f in batch_frames if f]
            if not batch_frames:
                video_semantics.extend([torch.zeros(1, self.hidden_dim, device=self.device) for _ in batch_paths])
                continue
            prompt = "从这个视频中提取意图相关线索：面部表情、语调、手势。输出：短描述。"
            inputs = self.processor(images=batch_frames, text=[prompt] * len(batch_frames), return_tensors="pt",
                                    padding=True).to(self.device)
            try:
                with torch.no_grad():
                    outputs = self.visual_llm(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1].mean(dim=1)
                    semantic_features = self.video_semantic_proj(last_hidden.float())
                video_semantics.extend(semantic_features)
            except Exception as e:
                self.logger.error(f"视频语义提取失败: {e}")
                video_semantics.extend([torch.zeros(1, self.hidden_dim, device=self.device) for _ in batch_paths])
        return torch.cat(video_semantics, dim=0) if video_semantics else None

    # 音频提取类似，省略以简洁

    def forward(self, V, A, video_paths, audio_paths=None):
        B, L, D = V.shape

        # 检查是否启用了MLLM
        if self.visual_llm is None:
            self.logger.info("MLLM未启用或加载失败，返回原始特征")
            return V, A

        # 检查是否有视频路径
        if video_paths is None or len(video_paths) == 0:
            self.logger.warning("MLLM启用但未提供视频路径，返回原始特征")
            return V, A

        # 提取视频语义特征
        video_semantics = self.extract_video_semantics(video_paths)

        if video_semantics is None:
            self.logger.warning("视频语义提取失败，返回原始特征")
            return V, A

        # 确保维度匹配
        if video_semantics.shape[0] != B:
            self.logger.warning(f"语义特征维度不匹配: video_semantics={video_semantics.shape}, V={V.shape}")
            # 尝试调整维度
            if video_semantics.shape[0] == 1 and B > 1:
                video_semantics = video_semantics.repeat(B, 1)
            else:
                return V, A

        # 特征融合
        V_mean = V.mean(dim=1)
        fusion_input = torch.cat([V_mean, video_semantics], dim=-1)
        gate = self.fusion_gate(fusion_input)
        semantics_expanded = video_semantics.unsqueeze(1).expand(-1, L, -1)
        modulated = gate.unsqueeze(1) * semantics_expanded * V
        enhancement = self.enhancer(modulated)
        V_enhanced = V + enhancement

        # 音频处理（如果启用）
        if self.use_audio and audio_paths is not None and self.audio_llm is not None:
            # 这里可以添加音频处理逻辑
            A_enhanced = A  # 暂时保持不变
        else:
            A_enhanced = A

        return V_enhanced, A_enhanced