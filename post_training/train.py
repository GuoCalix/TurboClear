import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None and torch.cuda.is_available():
    torch.cuda.set_device(int(local_rank))

from HYPIR.trainer.objectclear_fusion import ObjectClearFusionTrainer

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)


if config.base_model_type == "objectclear_fusion":
    trainer = ObjectClearFusionTrainer(config)
    trainer.run()
else:
    raise ValueError(
        "This release's post-training entry point only supports "
        f"base_model_type=objectclear_fusion, got {config.base_model_type!r}."
    )
