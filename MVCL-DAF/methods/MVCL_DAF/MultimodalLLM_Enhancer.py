import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor
import warnings
import cv2  # 用于视频帧提取
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')


class MultimodalLLM_Enhancer(nn.Module):
    """
    处理视频的多模态大模型增强模块，支持视觉和音频。
    """

    def __init__(self, args, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_dim = args.text_feat_dim  # 1024，与视频特征维度对齐
        self.use_audio = args.use_audio_mllm if hasattr(args, 'use_audio_mllm') else False  # 可配置是否处理音频

        # ========== 初始化 Qwen2.5-VL（视觉）==========
        print("加载 Qwen2.5-VL-7B-Instruct 用于视频语义理解...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True
            )
            self.visual_llm = AutoModelForVision2Seq.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            for param in self.visual_llm.parameters():
                param.requires_grad = False
            self.visual_llm.eval()
            print("Qwen2.5-VL 加载成功（仅视觉模式）")
        except Exception as e:
            print(f"Qwen2.5-VL 加载失败: {e}")
            self.visual_llm = None

        # ========== 初始化 Whisper（音频，如果需要）==========
        if self.use_audio:
            print("加载 Whisper-small 用于音频转录...")
            try:
                self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.audio_llm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.audio_llm.to(self.device)
                self.audio_llm.eval()
                print("Whisper 加载成功")
            except Exception as e:
                print(f"Whisper 加载失败: {e}")
                self.audio_llm = None

        # 视频语义投影
        self.video_semantic_proj = nn.Sequential(
            nn.Linear(4096, self.hidden_dim),  # Qwen隐藏层->你的维度
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 音频语义投影（如果启用）
        if self.use_audio:
            self.audio_semantic_proj = nn.Sequential(
                nn.Linear(1024, self.hidden_dim),  # Whisper隐藏层->你的维度
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )

        # 特征增强适配器
        self.enhancer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        print("多模态增强模块初始化完成")

    def load_video_frames(self, video_path, num_frames=3):
        """
        从MP4加载关键帧（PIL Image列表）
        """
        cap = cv2.VideoCapture(video_path)
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
        """
        批量提取视频语义特征 [B, hidden_dim]
        """
        if self.visual_llm is None:
            return None

        video_semantics = []
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i + batch_size]
            batch_frames = [self.load_video_frames(path) for path in batch_paths]
            batch_frames = [f for f in batch_frames if f]  # 过滤无效

            if not batch_frames:
                video_semantics.extend([torch.zeros(1, self.hidden_dim, device=self.device) for _ in batch_paths])
                continue

            prompt = """从这个视频中提取意图相关线索：面部表情、语调、手势。输出：短描述。"""

            inputs = self.processor(
                images=batch_frames,
                text=[prompt] * len(batch_frames),
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.visual_llm(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1].mean(dim=1)  # [batch, hidden_size]
                semantic_features = self.video_semantic_proj(last_hidden.float())

            video_semantics.extend(
                semantic_features.unsqueeze(0) if semantic_features.dim() == 1 else semantic_features)

        return torch.cat(video_semantics, dim=0) if video_semantics else None

    def extract_audio_semantics(self, audio_paths, batch_size=4):
        """
        批量提取音频语义（如果启用） [B, hidden_dim]
        """
        if not self.use_audio or self.audio_llm is None:
            return None

        # 类似视频的批量处理逻辑，省略细节（使用torchaudio加载音频波形，然后processor处理）
        # 返回音频语义特征

    def forward(self, V, A, video_paths, audio_paths=None):
        """
        V: 视觉特征 [B, L, D]
        A: 音频特征 [B, L, D]
        video_paths: List[str] MP4路径
        audio_paths: List[str] (可选)

        返回: 增强后的 V_enhanced, A_enhanced
        """
        B, L, D = V.shape

        # 提取语义
        video_semantics = self.extract_video_semantics(video_paths)
        audio_semantics = self.extract_audio_semantics(audio_paths) if self.use_audio else None

        if video_semantics is None:
            video_semantics = torch.zeros(B, self.hidden_dim, device=self.device)

        V_mean = V.mean(dim=1)
        fusion_input = torch.cat([V_mean, video_semantics], dim=-1)
        gate = self.fusion_gate(fusion_input)

        semantics_expanded = video_semantics.unsqueeze(1).expand(-1, L, -1)
        modulated = gate.unsqueeze(1) * semantics_expanded * V
        enhancement = self.enhancer(modulated)
        V_enhanced = V + enhancement

        # 音频类似（如果启用）
        A_enhanced = A  # 占位

        return V_enhanced, A_enhanced