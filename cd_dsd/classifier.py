import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


class MCDropoutClassifier(nn.Module):

    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.3,
                 pretrained: bool = True):
        super().__init__()

        backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.drop     = nn.Dropout(p=dropout_rate)
        self.fc       = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = self.pool(f).flatten(1)
        f = self.drop(f)
        return self.fc(f)

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, n_samples: int = 30) -> dict:
        self.eval()
        enable_dropout(self)

        probs_list = []
        for _ in range(n_samples):
            logits = self(x)
            probs_list.append(torch.sigmoid(logits))

        probs = torch.stack(probs_list, dim=1)
        mean_prob = probs.mean(dim=1)
        variance  = probs.var(dim=1)

        eps = 1e-7
        p   = mean_prob.clamp(eps, 1 - eps)
        entropy_per_class = -(p * p.log() + (1 - p) * (1 - p).log())
        entropy = entropy_per_class.mean(dim=1)

        return dict(mean_prob=mean_prob, variance=variance,
                    entropy=entropy, raw_probs=probs)

    @torch.no_grad()
    def uncertainty_scalar(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        return self.mc_predict(x, n_samples)["entropy"]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))
