"""
Spatial-Temporal 2D-CNN-BiGRU and Transductive CDAN Architecture
================================================================
Implements:
1. Spatial 2D-CNN: Extracts spatial cortical & hemispheric features from (5, 9, 9) grids per frame.
2. Temporal Bi-GRU: Models temporal affective dynamics over T frames with attentive temporal pooling.
3. 4-Class Emotion Classifier: Produces class logits and calibrated softmax probabilities.
4. Transductive CDAN Discriminator: Multilinear conditioning (f ⊗ g, 512D) with entropy weighting
   and dynamic Gradient Reversal Layer (GRL) annealing.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Gradient Reversal Layer (GRL).
    Forward pass acts as identity.
    Backward pass scales gradients by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SpatialCNN(nn.Module):
    """
    Lightweight 2D-CNN operating over (5, 9, 9) spatial EEG grids.
    Extracts local cortical field gradients and left-right hemispheric asymmetry.
    """
    def __init__(self, in_channels=5, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (Batch * Time, Channels=5, Height=9, Width=9)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class AttentiveTemporalPooling(nn.Module):
    """
    Self-attention pooling over the temporal sequence dimension T.
    Computes a learned weighted sum of temporal state representations.
    """
    def __init__(self, in_dim=128, hidden_dim=32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        # h: (B, T, in_dim)
        scores = self.attn(h) # (B, T, 1)
        weights = torch.softmax(scores, dim=1) # (B, T, 1)
        pooled = torch.sum(h * weights, dim=1) # (B, in_dim)
        return pooled, weights

class ConditionalDomainDiscriminator(nn.Module):
    """
    CDAN Domain Discriminator with Multilinear Conditioning (f ⊗ g).
    Conditioned on the tensor product of feature representations f (128D) and
    classifier probability distributions g (4D), yielding a 512D input space.
    """
    def __init__(self, feature_dim=128, num_classes=4, hidden_dim=256):
        super().__init__()
        input_dim = feature_dim * num_classes # 128 * 4 = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )

    def forward(self, f, g, alpha=0.0):
        # f: (B, feature_dim=128), g: (B, num_classes=4)
        # Multilinear outer product: (B, 4, 1) x (B, 1, 128) -> (B, 4, 128) -> (B, 512)
        op = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        op = op.view(f.size(0), -1)
        
        # Apply GRL
        rev_op = ReverseLayerF.apply(op, alpha)
        domain_logits = self.net(rev_op) # (B, 1)
        return domain_logits

class SpatialTemporalCDAN(nn.Module):
    """
    Unified Spatial-Temporal 2D-CNN-BiGRU + CDAN Architecture for SEED-IV EEG.
    """
    def __init__(self, in_channels=5, spatial_dim=128, rnn_hidden=64, num_classes=4):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        self.temporal_dim = rnn_hidden * 2 # Bidirectional = 128D
        self.num_classes = num_classes
        
        # 1. Spatial 2D-CNN
        self.spatial_net = SpatialCNN(in_channels=in_channels, out_dim=spatial_dim)
        
        # 2. Temporal Bi-GRU
        self.temporal_net = nn.GRU(
            input_size=spatial_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 3. Attentive Temporal Pooling
        self.temporal_pooling = AttentiveTemporalPooling(in_dim=self.temporal_dim, hidden_dim=32)
        
        # 4. Emotion Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 5. Conditional Domain Discriminator
        self.discriminator = ConditionalDomainDiscriminator(
            feature_dim=self.temporal_dim,
            num_classes=num_classes,
            hidden_dim=256
        )

    def extract_features(self, x):
        """
        Extracts pooled holistic temporal-spatial feature representations.
        Parameters:
            x: (B, T, 5, 9, 9)
        Returns:
            f: (B, 128)
            attn_weights: (B, T, 1)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        s_feat = self.spatial_net(x_flat) # (B*T, spatial_dim)
        s_feat = s_feat.view(B, T, self.spatial_dim) # (B, T, spatial_dim)
        
        t_out, _ = self.temporal_net(s_feat) # (B, T, temporal_dim)
        f, attn_weights = self.temporal_pooling(t_out) # (B, temporal_dim)
        return f, attn_weights

    def forward(self, x, alpha=0.0):
        """
        Forward pass yielding classification logits, probabilities, and domain logits.
        """
        f, attn_weights = self.extract_features(x)
        logits = self.classifier(f) # (B, 4)
        probs = torch.softmax(logits, dim=1) # (B, 4)
        domain_logits = self.discriminator(f, probs, alpha=alpha) # (B, 1)
        return logits, domain_logits, f, probs

    def compute_cdan_loss(self, source_x, source_y, target_x, alpha=1.0, w_dom=0.1):
        """
        Computes the joint CDAN objective with entropy-conditioned weighting.
        
        Loss = L_cls(source) + w_dom * [ L_dom(source, 1) + L_dom(target, 0) ]
        """
        criterion_cls = nn.CrossEntropyLoss()
        criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # 1. Source forward pass (Emotion classification + Domain discrimination)
        s_logits, s_dom_logits, s_f, s_probs = self.forward(source_x, alpha=alpha)
        loss_cls = criterion_cls(s_logits, source_y)
        
        # 2. Target forward pass (Domain discrimination only, labels 100% blind)
        t_logits, t_dom_logits, t_f, t_probs = self.forward(target_x, alpha=alpha)
        
        # 3. Entropy-Conditioned Weighting
        # H(g) = - sum_c g_c * log(g_c + eps)
        eps = 1e-8
        s_entropy = -torch.sum(s_probs * torch.log(s_probs + eps), dim=1)
        t_entropy = -torch.sum(t_probs * torch.log(t_probs + eps), dim=1)
        
        s_weight = 1.0 + torch.exp(-s_entropy)
        t_weight = 1.0 + torch.exp(-t_entropy)
        
        # Normalize weights so mean is 1.0 to preserve loss scale
        s_weight = s_weight / torch.mean(s_weight)
        t_weight = t_weight / torch.mean(t_weight)
        
        # 4. Domain Loss
        s_dom_labels = torch.ones_like(s_dom_logits) # 1.0 = Source
        t_dom_labels = torch.zeros_like(t_dom_logits) # 0.0 = Target
        
        s_dom_loss = torch.mean(s_weight * criterion_bce(s_dom_logits, s_dom_labels).squeeze(1))
        t_dom_loss = torch.mean(t_weight * criterion_bce(t_dom_logits, t_dom_labels).squeeze(1))
        loss_dom = 0.5 * (s_dom_loss + t_dom_loss)
        
        total_loss = loss_cls + w_dom * loss_dom
        
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_dom': loss_dom,
            's_logits': s_logits,
            't_logits': t_logits
        }

if __name__ == '__main__':
    print("Testing SpatialTemporalCDAN architecture...")
    model = SpatialTemporalCDAN()
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test batch forward
    src_x = torch.randn(8, 8, 5, 9, 9)
    src_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    tgt_x = torch.randn(8, 8, 5, 9, 9)
    
    loss_dict = model.compute_cdan_loss(src_x, src_y, tgt_x, alpha=1.0, w_dom=0.1)
    print("CDAN Loss computation:")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Classification Loss: {loss_dict['loss_cls'].item():.4f}")
    print(f"  Domain Adversarial Loss: {loss_dict['loss_dom'].item():.4f}")
    
    # Test backward pass
    loss_dict['total_loss'].backward()
    print("[PASS] Gradient backpropagation executed successfully without errors!")