import torch
from torch import nn


class ContextPriorEstimator(nn.Module):
    def __init__(self, num_emotions: int):
        super().__init__()
        self.num_emotions = num_emotions
        self.register_buffer('pi', torch.zeros(num_emotions))
        self.register_buffer('P', torch.zeros(num_emotions, 2))
        self.register_buffer('P_neg', torch.zeros(num_emotions, 2))
        self.register_buffer("q", torch.zeros(2))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit_from_loader(self, model, loader, device):
        model.eval()
        n = 0
        sum_y = torch.zeros(self.num_emotions, device=device)
        sum_joint = torch.zeros(self.num_emotions, 2, device=device)
        sum_q = torch.zeros(2, device=device)

        for batch in loader:
            x = batch['image'].to(device)
            y = batch['emotion'].to(device)
            out = model.forward_context(x)
            scene_conf = out['z_scene'].max(dim=1).values
            object_conf = out['z_object'].max(dim=1).values

            sum_y += y.sum(dim=0)
            sum_joint[:, 0] += (y * scene_conf.unsqueeze(1)).sum(dim=0)
            sum_joint[:, 1] += (y * object_conf.unsqueeze(1)).sum(dim=0)
            sum_q[0] += scene_conf.sum()
            sum_q[1] += object_conf.sum()
            n += y.shape[0]

        eps = 1e-6
        pi = (sum_y / max(n, 1)).clamp(0.0, 1.0)
        q = (sum_q / max(n, 1)).clamp(eps, 1.0 - eps)
        C = (sum_joint / max(n, 1)).clamp(0.0, 1.0)
        P = torch.stack([C[:, 0] / q[0], C[:, 1] / q[1]], dim=1).clamp(0.0, 1.0)
        P_neg = torch.stack([
            (pi - C[:, 0]) / (1.0 - q[0] + eps),
            (pi - C[:, 1]) / (1.0 - q[1] + eps),
        ], dim=1).clamp(0.0, 1.0)

        self.pi.copy_(pi.detach())
        self.q.copy_(q.detach())
        self.P.copy_(P.detach())
        self.P_neg.copy_(P_neg.detach())
        self.fitted.fill_(True)


class ProbabilisticFusion(nn.Module):
    def __init__(self, num_emotions: int, beta: float = 0.5, tau: float = 10.0, lambda_scale: float = 0.2):
        super().__init__()
        self.num_emotions = num_emotions
        self.beta = beta
        self.tau = tau
        self.lambda_scale = lambda_scale
        self.priors = ContextPriorEstimator(num_emotions)

    def forward(
            self,
            emotion_output: torch.Tensor,
            emotion_probs: torch.Tensor,
            z_scene: torch.Tensor,
            z_object: torch.Tensor,
    ):
        """
        Parameters
        ----------
        emotion_output:
            Raw 29-dimensional output of H_emotion, shape [B, 29].
            This is y^emotion in the manuscript and is modulated before MSE.
        emotion_probs:
            Precision/temperature-scaled emotion softmax, shape [B, 29].
            It is returned for inspection; it is not used as the MSE target.
        z_scene:
            Frozen Places-style 365-class scene soft pseudo-labels, shape [B, 365].
        z_object:
            Frozen COCO-style 80-class object soft pseudo-labels, shape [B, 80].

        Returns
        -------
        dict
            Fused priors and final 29-dimensional prediction.
        """
        if self.lambda_scale <= 0.0:
            raise ValueError("lambda_scale must be strictly positive.")

        batch_size = emotion_output.size(0)
        device = emotion_output.device
        dtype = emotion_output.dtype

        # Before fitting on the training split, use neutral priors only.
        # In real training, call fit_priors(train_loader) before the first epoch.
        if not bool(self.priors.fitted.item()):
            p_plus_global = torch.full(
                (self.num_emotions,),
                0.5,
                device=device,
                dtype=dtype,
            )
            p_minus_global = torch.full(
                (self.num_emotions,),
                0.5,
                device=device,
                dtype=dtype,
            )
        else:
            # Eq. (14): p_i^+ = max_j P^+_{i,j}, p_i^- = max_j P^-_{i,j}.
            p_plus_global = self.priors.P.to(
                device=device,
                dtype=dtype,
            ).max(dim=1).values

            p_minus_global = self.priors.P_neg.to(
                device=device,
                dtype=dtype,
            ).max(dim=1).values

        # Eq. (13): Q_i = sigmoid(alpha * (p_i^+ - tau)).
        # In this implementation:
        #   self.tau  = gate sharpness, default 10.0
        #   self.beta = gate threshold, default 0.5
        gate_global = torch.sigmoid(
            self.tau * (p_plus_global - self.beta)
        )

        # Eq. (15): p_hat_i = Q_i p_i^+ + (1 - Q_i) p_i^-.
        p_fused_global = (
                gate_global * p_plus_global
                + (1.0 - gate_global) * p_minus_global
        )

        # Eq. (16): y_tilde = (1 / lambda) * p_hat element-wise y_emotion.
        p_plus = p_plus_global.unsqueeze(0).expand(batch_size, -1)
        p_minus = p_minus_global.unsqueeze(0).expand(batch_size, -1)
        gate = gate_global.unsqueeze(0).expand(batch_size, -1)
        p_fused = p_fused_global.unsqueeze(0).expand(batch_size, -1)

        y_hat = (
                        p_fused / self.lambda_scale
                ) * emotion_output

        # Retained only as diagnostic information. z_scene and z_object
        # are used to fit priors on the training split, not to rescale
        # P+ or P- in Equation (14)-(16).
        scene_conf = z_scene.max(dim=1).values.unsqueeze(1)
        object_conf = z_object.max(dim=1).values.unsqueeze(1)

        return {
            "scene_conf": scene_conf,
            "object_conf": object_conf,
            "p_avail": p_plus,
            "p_nonavail": p_minus,
            "Q": gate,
            "p_fused": p_fused,
            "y_hat": y_hat,
            "emotion_probs": emotion_probs,
        }
