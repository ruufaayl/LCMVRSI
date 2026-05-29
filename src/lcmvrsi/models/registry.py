from __future__ import annotations

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.utils.registry import Registry

MODELS: Registry[SequenceModel] = Registry("model")
register_model = MODELS.register
get_model = MODELS.get
list_models = MODELS.names
