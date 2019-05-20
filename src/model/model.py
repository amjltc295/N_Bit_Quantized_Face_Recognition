import torch

from base import BaseModel
import model.backbone as backbones
import model.head as heads


class FaceRecognitionModel(BaseModel):
    def __init__(
        self, backbone, head,
        n_class, embedding_size=512, input_size=[112, 112],
        input_channel=32, device_id=[0], **kwargs
    ):
        super().__init__()
        if backbone in ['MobileNetV2', 'QuantizedMobileNetV2']:
            self.backbone = getattr(backbones, backbone)(n_class=embedding_size, input_channel=input_channel)
        elif backbone in ['QuantizedMobileNet']:
            self.backbone = getattr(backbones, backbone)(**kwargs)
        else:
            self.backbone = getattr(backbones, backbone)(input_size)
        self.head = getattr(heads, head)(in_features=embedding_size, out_features=n_class, device_id=device_id)

    def get_features(self, x):
        features = self.backbone(x)
        return features

    def forward(self, x, labels):
        features = self.get_features(x)
        out = self.head(features, labels)
        return out

    def load_backbone(self, checkpoint):
        state_dict = checkpoint['state_dict']
        used_state_dict = {}
        for k, v in state_dict.items():
            if 'backbone' in k:
                used_state_dict[k] = v

        self.load_state_dict(used_state_dict, strict=False)
