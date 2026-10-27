"""Check dynamically loaded Eagle code without allocating a model or using CUDA."""

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from gr00t.model.policy import Gr00tPolicy


def main():
    assert 'panda_omron' in DATA_CONFIG_MAP
    config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
    assert config.vision_config.model_type == 'siglip_vision_model'
    get_class_from_dynamic_module(config.auto_map['AutoModel'], DEFAULT_EAGLE_PATH)
    print('GR00T policy and dynamic Eagle imports passed (SigLIP, no FlashAttention).')


if __name__ == '__main__':
    main()
