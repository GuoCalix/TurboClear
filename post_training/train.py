import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None and torch.cuda.is_available():
    torch.cuda.set_device(int(local_rank))

from HYPIR.trainer.objectclear import ObjectClearTrainer
from HYPIR.trainer.objectclear_dmd import ObjectClearDMDTrainer
from HYPIR.trainer.objectclear_fusion import ObjectClearFusionTrainer
from HYPIR.trainer.objectclear_lcm import ObjectClearLCMTrainer
from HYPIR.trainer.objectclear_new import ObjectClearNewTrainer

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.config)


if config.base_model_type == "objectclear":
    trainer = ObjectClearTrainer(config)
    trainer.run()
elif config.base_model_type == "objectclear_new":
    trainer = ObjectClearNewTrainer(config)
    trainer.run()
elif config.base_model_type == "objectclear_lcm":
    trainer = ObjectClearLCMTrainer(config)
    trainer.run()
elif config.base_model_type == "objectclear_dmd":
    trainer = ObjectClearDMDTrainer(config)
    trainer.run()
elif config.base_model_type == "objectclear_fusion":
    trainer = ObjectClearFusionTrainer(config)
    trainer.run()
else:
    raise ValueError(f"Unsupported model type: {config.base_model_type}")
