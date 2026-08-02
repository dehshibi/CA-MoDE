from torch import nn
from .backbone import SharedFeatureEncoder
from .context_experts import SceneContextExpert, ObjectContextExpert, EmotionExpert
from .fusion import ProbabilisticFusion


class CAMoDE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedFeatureEncoder(out_channels=cfg.shared_channels)
        self.scene_expert = SceneContextExpert(cfg.shared_channels, cfg.num_scene_classes, cfg.kappa, cfg.dropout)
        self.object_expert = ObjectContextExpert(cfg.shared_channels, cfg.num_object_classes, cfg.kappa, cfg.dropout)
        self.emotion_expert = EmotionExpert(cfg.shared_channels, cfg.num_emotions, cfg.dropout)
        self.fusion = ProbabilisticFusion(cfg.num_emotions, cfg.gate_threshold, cfg.gate_sharpness, cfg.lambda_scale)
        self.freeze_context_experts()

    def freeze_context_experts(self):
        for p in self.scene_expert.tower.parameters():
            p.requires_grad = False
        for p in self.scene_expert.classifier.parameters():
            p.requires_grad = False
        for p in self.object_expert.tower.parameters():
            p.requires_grad = False
        for p in self.object_expert.classifier.parameters():
            p.requires_grad = False

    def forward_context(self, x):
        feat = self.encoder(x)
        z_scene, y_scene = self.scene_expert(feat)
        z_object, y_object = self.object_expert(feat)
        emotion_logits, emotion_probs = self.emotion_expert(feat)
        return {
            'feat': feat,
            'z_scene': z_scene,
            'y_scene': y_scene,
            'z_object': z_object,
            'y_object': y_object,
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
        }

    def forward(self, x):
        ctx = self.forward_context(x)
        fused = self.fusion(
            ctx['emotion_logits'],
            ctx['emotion_probs'],
            ctx['z_scene'],
            ctx['z_object'],
        )
        ctx.update(fused)
        return ctx
