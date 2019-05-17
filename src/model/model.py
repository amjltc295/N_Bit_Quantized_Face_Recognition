from base import BaseModel
import model.backbone as backbones
import model.head as heads


class FaceRecognitionModel(BaseModel):
    def __init__(self, backbone, head, n_class=1000, embedding_size=512, input_size=112, device_id=[0], **kwargs):
        super().__init__()
        if backbone in ['MobileNetV2']:
            self.backbone = getattr(backbones, backbone)(embedding_size, input_size)
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
