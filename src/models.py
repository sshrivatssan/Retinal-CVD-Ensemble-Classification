import timm
import torch
import torch.nn as nn


def build_baseline_model(model_name: str, num_classes: int = 2):
    """Create a pretrained baseline CNN classifier."""

    if model_name == "densenet121":
        return timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=num_classes
        )

    if model_name == "efficientnet_b3":
        return timm.create_model(
            "tf_efficientnet_b3_ns",
            pretrained=True,
            num_classes=num_classes
        )

    if model_name == "mobilenetv3_large":
        return timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=num_classes
        )

    raise ValueError(f"Unsupported model: {model_name}")


class FusionDenseNet(nn.Module):
    """DenseNet121 fusion model using handcrafted texture features."""

    def __init__(self, hf_dim, num_classes=2, p_drop=0.2):
        super().__init__()

        self.backbone = timm.create_model(
            "densenet121",
            pretrained=True,
            num_classes=0
        )

        self.feat_dim = self.backbone.num_features

        self.bn_img = nn.BatchNorm1d(self.feat_dim)
        self.bn_hf = nn.BatchNorm1d(hf_dim)

        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(self.feat_dim + hf_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_hf):
        z = self.backbone(x_img)
        z = self.bn_img(z)

        h = self.bn_hf(x_hf)

        out = torch.cat([z, h], dim=1)

        return self.head(out)


class FusionEffB3(nn.Module):
    """EfficientNet-B3 fusion model using handcrafted texture features."""

    def __init__(self, hf_dim, num_classes=2, p_drop=0.2):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b3_ns",
            pretrained=True,
            num_classes=0
        )

        self.feat_dim = self.backbone.num_features

        self.img_proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop)
        )

        self.hf_proj = nn.Sequential(
            nn.LayerNorm(hf_dim),
            nn.Linear(hf_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop)
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_hf):
        z = self.backbone(x_img)
        z = self.img_proj(z)

        h = self.hf_proj(x_hf)

        return self.head(
            torch.cat([z, h], dim=1)
        )


class FusionMobileNetV3(nn.Module):
    """MobileNetV3-Large fusion model using handcrafted texture features."""

    def __init__(self, hf_dim, num_classes=2, p_drop=0.2):
        super().__init__()

        self.backbone = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                3,
                224,
                224
            )

            feat_dim = self.backbone(
                dummy
            ).shape[1]

        self.feat_dim = feat_dim

        self.img_proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop)
        )

        self.hf_proj = nn.Sequential(
            nn.LayerNorm(hf_dim),
            nn.Linear(hf_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop)
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_hf):
        z = self.backbone(x_img)
        z = self.img_proj(z)

        h = self.hf_proj(x_hf)

        return self.head(
            torch.cat([z, h], dim=1)
        )


def build_fusion_model(
    model_name: str,
    hf_dim: int,
    num_classes: int = 2,
    p_drop: float = 0.2
):
    """Create the requested handcrafted-feature fusion model."""

    if model_name == "densenet121":
        return FusionDenseNet(
            hf_dim=hf_dim,
            num_classes=num_classes,
            p_drop=p_drop
        )

    if model_name == "efficientnet_b3":
        return FusionEffB3(
            hf_dim=hf_dim,
            num_classes=num_classes,
            p_drop=p_drop
        )

    if model_name == "mobilenetv3_large":
        return FusionMobileNetV3(
            hf_dim=hf_dim,
            num_classes=num_classes,
            p_drop=p_drop
        )

    raise ValueError(
        f"Unsupported fusion model: {model_name}"
    )