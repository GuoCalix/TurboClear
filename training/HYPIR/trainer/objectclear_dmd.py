import csv
import gc
import json
import os
import random
import shutil
from contextlib import contextmanager
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import DistributedType
from diffusers import UNet2DConditionModel
from peft import LoraConfig
from PIL import Image
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from HYPIR.dataset.sd_inpaint_dataset import SDInpaintImageDataset
from HYPIR.trainer.objectclear import ObjectClearTrainer
from HYPIR.utils.attention_guided_fusion import attention_guided_fusion
from HYPIR.dataset.ober import (
    OBERDirectoryDataset,
    extract_state_dict,
    normalize_state_dict_keys,
    resolve_state_file,
)
from HYPIR.utils.objectclear_helpers import (
    BalancedL1Loss,
    clear_cross_attention_scores,
    get_object_localization_loss,
    resize_attn_map_divide2,
    unet_store_cross_attention_scores,
)
from HYPIR.utils.common import print_vram_state
from HYPIR.utils.ema import EMAModel

logger = get_logger(__name__, log_level="INFO")


def _rank_aware_worker_init_fn(worker_id):
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    worker_seed = (torch.initial_seed() + rank * 1000003) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class ObjectClearDMDTrainer(ObjectClearTrainer):
    """ObjectClear inpainting DMD trainer with independent real/fake score UNets."""

    def _trainable_unet_dtype(self):
        if bool(getattr(self.config, "fft_trainable_unet_fp32", False)):
            return torch.float32
        return self.weight_dtype

    def _dmd_enabled(self):
        return bool(getattr(self.config, "use_DMD", True)) and not bool(
            getattr(self.config, "pretrain_lpips_only", False)
        )

    def _free_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_transient_cuda_tensors(self):
        for scores_name in ("cross_attention_scores", "teacher_cross_attention_scores"):
            scores = getattr(self, scores_name, None)
            if isinstance(scores, dict):
                scores.clear()
        for attr in ("G_pred", "G_pred_latents", "batch_inputs", "last_generator_attention_map"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.last_dmd_debug = {}
        self._free_cuda_cache()

    def _fake_weight_probe_enabled(self):
        return bool(getattr(self.config, "debug_fake_weight_probe", False))

    def _fake_weight_probe(self, max_tensors=None):
        if not self._fake_weight_probe_enabled() or not hasattr(self, "fake_G"):
            return {}
        if max_tensors is None:
            max_tensors = int(getattr(self.config, "debug_fake_weight_probe_tensors", 16))
        model = self.unwrap_model(self.fake_G)
        checksum = torch.tensor(0.0, device=self.device)
        abs_mean = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for _, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                data = param.detach().float()
                checksum = checksum + data.flatten()[: min(1024, data.numel())].sum().to(self.device)
                abs_mean = abs_mean + data.abs().mean().to(self.device)
                count = count + 1.0
                if int(count.item()) >= max_tensors:
                    break
        if count.item() == 0:
            return {}
        return {
            "fake_probe_checksum": checksum.detach(),
            "fake_probe_abs_mean": (abs_mean / count.clamp_min(1.0)).detach(),
        }

    def _fake_trainable_param_slice(self):
        if not self._fake_weight_probe_enabled() or not hasattr(self, "fake_G"):
            return None
        model = self.unwrap_model(self.fake_G)
        for param in model.parameters():
            if param.requires_grad:
                return param
        return None

    def _fake_grad_probe(self):
        if not self._fake_weight_probe_enabled() or not hasattr(self, "fake_params"):
            return {}
        total_sq = torch.tensor(0.0, device=self.device)
        max_abs = torch.tensor(0.0, device=self.device)
        grad_count = torch.tensor(0.0, device=self.device)
        none_count = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for param in self.fake_params:
                grad = getattr(param, "grad", None)
                if grad is None:
                    none_count = none_count + 1.0
                    continue
                grad_f = grad.detach().float()
                total_sq = total_sq + grad_f.pow(2).sum().to(self.device)
                max_abs = torch.maximum(max_abs, grad_f.abs().amax().to(self.device))
                grad_count = grad_count + 1.0
        lr = 0.0
        if hasattr(self, "fake_opt") and getattr(self.fake_opt, "param_groups", None):
            lr = float(self.fake_opt.param_groups[0].get("lr", 0.0))
        probe = {
            "fake_probe_grad_norm": total_sq.sqrt().detach(),
            "fake_probe_grad_max": max_abs.detach(),
            "fake_probe_grad_param_count": grad_count.detach(),
            "fake_probe_grad_none_count": none_count.detach(),
            "fake_probe_optimizer_lr": torch.tensor(lr, device=self.device),
        }
        first_param = None
        for param in self.fake_params:
            if param.requires_grad:
                first_param = param
                break
        if first_param is not None:
            with torch.no_grad():
                data = first_param.detach().flatten()[:1024]
                data_f = data.float()
                dtype_code = 0.0
                if data.dtype == torch.float32:
                    dtype_code = 32.0
                elif data.dtype == torch.float16:
                    dtype_code = 16.0
                elif data.dtype == torch.bfloat16:
                    dtype_code = 17.0
                else:
                    dtype_code = -1.0
                try:
                    next_up = torch.nextafter(data, torch.full_like(data, float("inf"))).float()
                    ulp = (next_up - data_f).abs()
                    finite_ulp = ulp[torch.isfinite(ulp) & (ulp > 0)]
                    if finite_ulp.numel() > 0:
                        ulp_mean = finite_ulp.mean()
                        ulp_min = finite_ulp.min()
                    else:
                        ulp_mean = torch.tensor(0.0, device=data.device)
                        ulp_min = torch.tensor(0.0, device=data.device)
                except Exception:
                    ulp_mean = torch.tensor(0.0, device=data.device)
                    ulp_min = torch.tensor(0.0, device=data.device)
                probe.update(
                    {
                        "fake_probe_param_dtype_code": torch.tensor(dtype_code, device=self.device),
                        "fake_probe_param_abs_mean": data_f.abs().mean().to(self.device),
                        "fake_probe_param_ulp_mean": ulp_mean.to(self.device),
                        "fake_probe_param_ulp_min": ulp_min.to(self.device),
                        "fake_probe_lr_over_ulp_mean": torch.tensor(lr, device=self.device)
                        / (ulp_mean.to(self.device) + 1e-30),
                    }
                )
        return probe

    def init_models(self):
        if bool(getattr(self.config, "use_external_D", False)):
            self.config.use_D = True
        elif self._dmd_enabled() and bool(getattr(self.config, "use_gan", False)):
            self.config.use_D = False
        ObjectClearTrainer.init_models(self)
        self.cross_attention_scores = {}
        self.teacher_cross_attention_scores = {}
        self.object_localization_loss_fn = None

        self._load_generator_weights()
        logger.info(f"Use DMD: {self._dmd_enabled()}")
        if self._dmd_enabled():
            self.init_teacher_generator()
            self.init_fake_generator()
            self.init_fake_classifier()
            self._load_fft_fake_classifier_weights()

        self._train_attention_hook_enabled = (
            getattr(self.config, "object_localization", False)
            or getattr(self.config, "apply_attention_guided_fusion", False)
            or getattr(self.config, "debug_save_attention_map", False)
        )
        logger.info(f"Use localization: {getattr(self.config, 'object_localization', False)}")
        logger.info(f"Use attention fusion: {getattr(self.config, 'apply_attention_guided_fusion', False)}")
        logger.info(
            f"Use validation attention fusion: "
            f"{getattr(self.config, 'apply_attention_guided_fusion_in_validation', False)}"
        )
        if self._train_attention_hook_enabled:
            unet_store_cross_attention_scores(
                self.G,
                self.cross_attention_scores,
                layers=getattr(self.config, "attn_loss_layers", 5),
            )
        if getattr(self.config, "object_localization", False):
            self.object_localization_loss_fn = BalancedL1Loss(
                threshold=getattr(self.config, "object_localization_threshold", 0.1),
                background_loss_weight=getattr(self.config, "background_loss_weight", 1.0),
            )

    def _restore_attention_score_hooks(self, model):
        for module in self.unwrap_model(model).modules():
            saved_processor = getattr(module, "_dmd_saved_attn_processor", None)
            if saved_processor is not None and hasattr(module, "set_processor"):
                module.set_processor(saved_processor)
                delattr(module, "_dmd_saved_attn_processor")
            old_get_attention_scores = getattr(module, "old_get_attention_scores", None)
            if old_get_attention_scores is None:
                continue
            module.get_attention_scores = old_get_attention_scores
            for attr in ("old_get_attention_scores", "new_get_attention_scores"):
                if hasattr(module, attr):
                    delattr(module, attr)

    def _enable_validation_attention_hook_if_needed(self):
        if not bool(getattr(self.config, "apply_attention_guided_fusion_in_validation", False)):
            return False
        if bool(getattr(self, "_train_attention_hook_enabled", False)):
            return False
        model = self.unwrap_model(self.G)
        for module in model.modules():
            if hasattr(module, "processor") and hasattr(module, "set_processor"):
                module._dmd_saved_attn_processor = module.processor
        unet_store_cross_attention_scores(
            model,
            self.cross_attention_scores,
            layers=getattr(self.config, "attn_loss_layers", 5),
        )
        return True

    def _load_unet_weights(self, model, weight_path, role, strict=False):
        state_file = resolve_state_file(weight_path)
        if state_file is None:
            raise FileNotFoundError(f"Could not resolve {role} checkpoint from: {weight_path}")

        ckpt_obj = torch.load(state_file, map_location="cpu")
        state_dict = extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported {role} checkpoint format: {state_file}")
        state_dict = normalize_state_dict_keys(state_dict)

        if strict:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            filtered = state_dict
        else:
            model_state = model.state_dict()
            filtered = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            missing, unexpected = model.load_state_dict(filtered, strict=False)
        logger.info(
            f"Loaded {role} weights from {state_file}: "
            f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def _load_generator_weights(self):
        if self._fft_enabled():
            return
        generator_path = getattr(self.config, "generator_checkpoint_path", None)
        if generator_path:
            self._load_unet_weights(
                self.G,
                generator_path,
                "generator",
                strict=bool(getattr(self.config, "strict_lora_load", False)),
            )

    def init_generator(self):
        if not self._needs_fft_init_checkpoint():
            if self._fft_enabled():
                self.G = UNet2DConditionModel.from_pretrained(
                    self.config.base_model_path,
                    subfolder="unet",
                    torch_dtype=self._trainable_unet_dtype(),
                ).to(self.device)
                self.G.train().requires_grad_(True)
                if self.config.gradient_checkpointing:
                    self.G.enable_gradient_checkpointing()
                logger.info("Initialized FFT generator from base model with full UNet training enabled.")
                return
            return ObjectClearTrainer.init_generator(self)

        merged_model_path = self._ensure_fft_merged_model()
        self.G = UNet2DConditionModel.from_pretrained(
            merged_model_path,
            subfolder="unet",
            torch_dtype=self._trainable_unet_dtype(),
        ).to(self.device)
        self.G.train().requires_grad_(True)
        if self.config.gradient_checkpointing:
            self.G.enable_gradient_checkpointing()
        logger.info(f"Initialized FFT generator from merged model: {merged_model_path}")

    def init_teacher_generator(self):
        self.teacher_G = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.teacher_G.eval().requires_grad_(False)

        teacher_path = getattr(self.config, "teacher_checkpoint_path", None)
        if teacher_path is None:
            logger.info("No teacher_checkpoint_path set; using base_model_path UNet as frozen DMD teacher.")
        else:
            lora_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=self.config.lora_modules,
            )
            self.teacher_G.add_adapter(lora_cfg)
            self.teacher_G.eval().requires_grad_(False)
            self._load_unet_weights(
                self.teacher_G,
                teacher_path,
                "teacher",
                strict=bool(getattr(self.config, "strict_lora_load", False)),
            )

        if getattr(self.config, "teacher_apply_attention_guided_fusion", False):
            unet_store_cross_attention_scores(
                self.teacher_G,
                self.teacher_cross_attention_scores,
                layers=getattr(self.config, "attn_loss_layers", 5),
            )

    def _add_lora_to_unet(self, model, role):
        model.eval().requires_grad_(False)
        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=self.config.lora_modules,
        )
        model.add_adapter(lora_cfg)
        lora_params = [p for p in model.parameters() if p.requires_grad]
        if not lora_params:
            raise ValueError(f"Failed to find LoRA parameters for {role}.")
        for param in lora_params:
            param.data = param.to(torch.float32)
        logger.info(f"Add lora parameters to {role}: {self.config.lora_modules}")

    def _resolve_optional_fake_checkpoint_path(self):
        explicit = getattr(self.config, "fake_generator_checkpoint_path", None)
        if explicit:
            return explicit

        checkpoint_path = getattr(self.config, "generator_checkpoint_path", None)
        if not checkpoint_path:
            return None

        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if os.path.isdir(checkpoint_path):
            candidate = os.path.join(checkpoint_path, "fake_state_dict.pth")
        else:
            candidate = os.path.join(os.path.dirname(checkpoint_path), "fake_state_dict.pth")
        return candidate if os.path.exists(candidate) else None

    def _load_fake_generator_weights(self):
        if self._fft_enabled():
            return
        fake_path = self._resolve_optional_fake_checkpoint_path()
        if fake_path:
            self._load_unet_weights(
                self.fake_G,
                fake_path,
                "fake score model",
                strict=bool(getattr(self.config, "strict_lora_load", False)),
            )

    def init_fake_generator(self):
        if self._needs_fft_init_checkpoint():
            merged_model_path = self._ensure_fft_merged_model()
            fake_unet_dir = os.path.join(merged_model_path, "fake_unet")
            if os.path.isdir(fake_unet_dir):
                self.fake_G = UNet2DConditionModel.from_pretrained(
                    merged_model_path,
                    subfolder="fake_unet",
                    torch_dtype=self._trainable_unet_dtype(),
                ).to(self.device)
                logger.info(f"Initialized FFT fake score model from merged fake UNet: {fake_unet_dir}")
            else:
                logger.warning("FFT merged fake_unet is missing; initializing fake score model from base UNet.")
                self.fake_G = UNet2DConditionModel.from_pretrained(
                    self.config.base_model_path,
                    subfolder="unet",
                    torch_dtype=self._trainable_unet_dtype(),
                ).to(self.device)
            self.fake_G.requires_grad_(True)
        else:
            self.fake_G = UNet2DConditionModel.from_pretrained(
                self.config.base_model_path,
                subfolder="unet",
                torch_dtype=self._trainable_unet_dtype() if self._fft_enabled() else self.weight_dtype,
            ).to(self.device)
            if self._fft_enabled():
                self.fake_G.requires_grad_(True)
            else:
                self._add_lora_to_unet(self.fake_G, "fake score model")
                self._load_fake_generator_weights()
        self.fake_G.train()
        if self.config.gradient_checkpointing:
            self.fake_G.enable_gradient_checkpointing()

    def _fft_enabled(self):
        return bool(getattr(self.config, "use_FFT", False))

    def _needs_fft_init_checkpoint(self):
        return self._fft_enabled() and bool(getattr(self.config, "generator_checkpoint_path", None))

    def _fft_merged_model_dir(self):
        configured = getattr(self.config, "fft_merged_model_dir", None)
        if configured:
            return os.path.abspath(os.path.expanduser(str(configured)))
        return os.path.join(os.path.abspath(str(self.config.output_dir)), "checkpoint-0")

    def _base_fft_init_enabled(self):
        return not self._dmd_enabled() and not getattr(self.config, "generator_checkpoint_path", None)

    def _full_generator_fft_init_enabled(self):
        checkpoint_path = getattr(self.config, "generator_checkpoint_path", None)
        if not checkpoint_path:
            return False
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if os.path.isdir(checkpoint_path):
            if not self._dmd_enabled():
                return os.path.exists(os.path.join(checkpoint_path, "state_dict.pth"))
            return os.path.exists(os.path.join(checkpoint_path, "state_dict.pth")) and not os.path.exists(
                os.path.join(checkpoint_path, "fake_state_dict.pth")
            )
        return os.path.isfile(checkpoint_path)

    def _ensure_fft_merged_model(self):
        merged_dir = self._fft_merged_model_dir()
        unet_dir = os.path.join(merged_dir, "unet")
        fake_unet_dir = os.path.join(merged_dir, "fake_unet")
        report_path = os.path.join(merged_dir, "merge_report.json")
        if self.config.resume_from_checkpoint:
            resume_name = os.path.basename(str(self.config.resume_from_checkpoint).rstrip("/"))
            if resume_name == "checkpoint-0":
                raise ValueError("checkpoint-0 is the FFT merged initialization model, not an accelerator resume checkpoint.")

        if self.accelerator.is_main_process:
            if self._base_fft_init_enabled():
                if os.path.isdir(unet_dir):
                    if self._base_fft_init_matches_config(report_path):
                        logger.info(f"Reusing base FFT initialization at {merged_dir}")
                    elif bool(getattr(self.config, "fft_merge_overwrite", False)):
                        logger.warning(f"Overwriting stale base FFT initialization at {merged_dir}")
                        shutil.rmtree(merged_dir)
                        self._run_base_fft_init(merged_dir)
                    else:
                        raise ValueError(
                            f"FFT initialization already exists but does not match current base config: {merged_dir}. "
                            "Use a new output_dir or set fft_merge_overwrite: true."
                        )
                else:
                    self._run_base_fft_init(merged_dir)
            elif self._full_generator_fft_init_enabled():
                if os.path.isdir(unet_dir):
                    if self._full_generator_fft_init_matches_config(report_path):
                        logger.info(f"Reusing full generator FFT initialization at {merged_dir}")
                    elif bool(getattr(self.config, "fft_merge_overwrite", False)):
                        logger.warning(f"Overwriting stale full generator FFT initialization at {merged_dir}")
                        shutil.rmtree(merged_dir)
                        self._run_full_generator_fft_init(merged_dir)
                    else:
                        raise ValueError(
                            f"FFT initialization already exists but does not match current generator checkpoint: {merged_dir}. "
                            "Use a new output_dir or set fft_merge_overwrite: true."
                        )
                else:
                    self._run_full_generator_fft_init(merged_dir)
            elif os.path.isdir(unet_dir) and os.path.isdir(fake_unet_dir):
                if self._fft_merge_matches_config(report_path):
                    logger.info(f"Reusing existing FFT merged model at {merged_dir}")
                elif bool(getattr(self.config, "fft_merge_overwrite", False)):
                    logger.warning(f"Overwriting stale FFT merged model at {merged_dir}")
                    shutil.rmtree(merged_dir)
                    self._run_fft_merge(merged_dir)
                else:
                    raise ValueError(
                        f"FFT merged model already exists but does not match current base/lora config: {merged_dir}. "
                        "Use a new output_dir or set fft_merge_overwrite: true."
                    )
            else:
                self._run_fft_merge(merged_dir)
        self.accelerator.wait_for_everyone()
        if not os.path.isdir(unet_dir):
            raise FileNotFoundError(f"FFT merged UNet was not created: {unet_dir}")
        if (
            self._dmd_enabled()
            and
            not self._base_fft_init_enabled()
            and not self._full_generator_fft_init_enabled()
            and not os.path.isdir(fake_unet_dir)
        ):
            raise FileNotFoundError(f"FFT merged fake UNet was not created: {fake_unet_dir}")
        return merged_dir

    def _base_fft_init_matches_config(self, report_path):
        if not os.path.exists(report_path):
            return False
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            return False
        base = os.path.abspath(os.path.expanduser(str(self.config.base_model_path)))
        report_base = os.path.abspath(os.path.expanduser(str(report.get("base_model_path", ""))))
        return report.get("init_type") == "base_unet" and report_base == base

    def _run_base_fft_init(self, merged_dir):
        base_model_path = os.path.abspath(os.path.expanduser(str(self.config.base_model_path)))
        source_unet_dir = os.path.join(base_model_path, "unet")
        if not os.path.isdir(source_unet_dir):
            raise FileNotFoundError(f"Base model UNet folder is missing: {source_unet_dir}")
        os.makedirs(merged_dir, exist_ok=True)
        target_unet_dir = os.path.join(merged_dir, "unet")
        if os.path.exists(target_unet_dir):
            shutil.rmtree(target_unet_dir)
        logger.info(f"Copying base UNet for FFT pretrain initialization: {source_unet_dir} -> {target_unet_dir}")
        shutil.copytree(source_unet_dir, target_unet_dir)
        summary = {
            "init_type": "base_unet",
            "base_model_path": base_model_path,
            "unet": target_unet_dir,
        }
        with open(os.path.join(merged_dir, "merge_report.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def _full_generator_fft_init_matches_config(self, report_path):
        if not os.path.exists(report_path):
            return False
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            return False
        base = os.path.abspath(os.path.expanduser(str(self.config.base_model_path)))
        checkpoint = os.path.abspath(os.path.expanduser(str(getattr(self.config, "generator_checkpoint_path", ""))))
        report_base = os.path.abspath(os.path.expanduser(str(report.get("base_model_path", ""))))
        report_checkpoint = os.path.abspath(os.path.expanduser(str(report.get("generator_checkpoint_path", ""))))
        return report.get("init_type") == "full_generator" and report_base == base and report_checkpoint == checkpoint

    def _run_full_generator_fft_init(self, merged_dir):
        checkpoint_path = os.path.abspath(os.path.expanduser(str(self.config.generator_checkpoint_path)))
        state_file = resolve_state_file(checkpoint_path)
        if state_file is None:
            raise FileNotFoundError(f"Could not resolve generator checkpoint from: {checkpoint_path}")

        model = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        )
        ckpt_obj = torch.load(state_file, map_location="cpu")
        state_dict = extract_state_dict(ckpt_obj)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported generator checkpoint format: {state_file}")
        state_dict = normalize_state_dict_keys(state_dict)
        model_state = model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if len(filtered) == 0:
            raise ValueError(f"No UNet weights from {state_file} matched base model {self.config.base_model_path}.")

        os.makedirs(merged_dir, exist_ok=True)
        target_unet_dir = os.path.join(merged_dir, "unet")
        if os.path.exists(target_unet_dir):
            shutil.rmtree(target_unet_dir)
        logger.info(
            f"Saving full generator FFT initialization to {target_unet_dir}: "
            f"filtered={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        model.save_pretrained(
            target_unet_dir,
            safe_serialization=bool(getattr(self.config, "fft_merge_safe_serialization", True)),
        )
        del model, ckpt_obj, state_dict, filtered

        summary = {
            "init_type": "full_generator",
            "base_model_path": os.path.abspath(os.path.expanduser(str(self.config.base_model_path))),
            "generator_checkpoint_path": checkpoint_path,
            "state_file": os.path.abspath(os.path.expanduser(str(state_file))),
            "unet": target_unet_dir,
        }
        with open(os.path.join(merged_dir, "merge_report.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def _fft_merge_matches_config(self, report_path):
        if not os.path.exists(report_path):
            return False
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception:
            return False
        base = os.path.abspath(os.path.expanduser(str(self.config.base_model_path)))
        student_lora = self._resolve_fft_checkpoint_file("state_dict.pth")
        fake_lora = self._resolve_fft_checkpoint_file("fake_state_dict.pth")
        report_base = os.path.abspath(os.path.expanduser(str(report.get("base_model_path", ""))))
        report_student = os.path.abspath(os.path.expanduser(str(report.get("student_lora_checkpoint", ""))))
        report_fake = os.path.abspath(os.path.expanduser(str(report.get("fake_lora_checkpoint", ""))))
        return report_base == base and report_student == student_lora and report_fake == fake_lora

    def _resolve_fft_checkpoint_file(self, filename):
        checkpoint_path = getattr(self.config, "generator_checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("use_FFT requires generator_checkpoint_path to point to a DMD LoRA checkpoint.")
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if os.path.isdir(checkpoint_path):
            resolved = os.path.join(checkpoint_path, filename)
        elif filename == "state_dict.pth":
            resolved = checkpoint_path
        else:
            resolved = os.path.join(os.path.dirname(checkpoint_path), filename)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Required FFT source checkpoint file is missing: {resolved}")
        return resolved

    def _run_fft_merge(self, merged_dir):
        student_lora_path = self._resolve_fft_checkpoint_file("state_dict.pth")
        fake_lora_path = self._resolve_fft_checkpoint_file("fake_state_dict.pth")
        logger.info(
            "Merging DMD LoRA checkpoints into full UNets for FFT initialization: "
            f"student={student_lora_path}, fake={fake_lora_path}, output={merged_dir}"
        )
        from tools.merge_objectclear_lora import merge_lora

        common_kwargs = dict(
            base_model_path=str(self.config.base_model_path),
            output_dir=str(merged_dir),
            lora_rank=int(getattr(self.config, "lora_rank", 0)) or None,
            lora_alpha=float(getattr(self.config, "lora_rank", 0)) or None,
            scale=getattr(self.config, "fft_merge_scale", None),
            save_dtype=str(getattr(self.config, "fft_merge_save_dtype", "fp32")),
            max_shard_size=str(getattr(self.config, "fft_merge_max_shard_size", "5GB")),
            safe_serialization=bool(getattr(self.config, "fft_merge_safe_serialization", True)),
            link_base_components=bool(getattr(self.config, "fft_merge_link_base_components", True)),
            copy_base_components=bool(getattr(self.config, "fft_merge_copy_base_components", False)),
            overwrite_components=True,
            overwrite=True,
            dry_run=False,
            strict=bool(getattr(self.config, "fft_merge_strict", True)),
            local_files_only=True,
        )
        student_report = merge_lora(
            Namespace(
                lora_checkpoint=str(student_lora_path),
                unet_subfolder="unet",
                output_unet_subfolder="unet",
                **common_kwargs,
            )
        )
        student_report_path = os.path.join(merged_dir, "merge_report_student.json")
        with open(student_report_path, "w", encoding="utf-8") as handle:
            json.dump(self._jsonify_merge_report(student_report), handle, indent=2, ensure_ascii=False)

        fake_report = merge_lora(
            Namespace(
                lora_checkpoint=str(fake_lora_path),
                unet_subfolder="unet",
                output_unet_subfolder="fake_unet",
                **common_kwargs,
            )
        )
        fake_report_path = os.path.join(merged_dir, "merge_report_fake.json")
        with open(fake_report_path, "w", encoding="utf-8") as handle:
            json.dump(self._jsonify_merge_report(fake_report), handle, indent=2, ensure_ascii=False)

        self._copy_fft_fake_classifier(merged_dir)
        summary = {
            "base_model_path": os.path.abspath(os.path.expanduser(str(self.config.base_model_path))),
            "student_lora_checkpoint": os.path.abspath(os.path.expanduser(str(student_lora_path))),
            "fake_lora_checkpoint": os.path.abspath(os.path.expanduser(str(fake_lora_path))),
            "fake_cls_head": os.path.join(merged_dir, "fake_cls_head.pth")
            if os.path.exists(os.path.join(merged_dir, "fake_cls_head.pth"))
            else None,
            "student_report": student_report_path,
            "fake_report": fake_report_path,
        }
        with open(os.path.join(merged_dir, "merge_report.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    def _jsonify_merge_report(self, report):
        if hasattr(report, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(report)
        return report

    def _copy_fft_fake_classifier(self, merged_dir):
        checkpoint_path = getattr(self.config, "generator_checkpoint_path", None)
        if not checkpoint_path:
            return
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        checkpoint_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        src = os.path.join(checkpoint_dir, "fake_cls_head.pth")
        dst = os.path.join(merged_dir, "fake_cls_head.pth")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied FFT fake classifier head initialization: {src} -> {dst}")
        elif self._internal_gan_enabled():
            logger.warning(f"fake_cls_head.pth is missing in FFT source checkpoint; classifier head will be random: {src}")

    def init_fake_classifier(self):
        if not self._internal_gan_enabled():
            return

        channels = int(self.fake_G.config.block_out_channels[-1])
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.fake_cls_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 1, kernel_size=1),
        ).to(self.device)
        self.fake_cls_head.train().requires_grad_(True)

    def _load_fft_fake_classifier_weights(self):
        if not self._needs_fft_init_checkpoint() or not self._internal_gan_enabled() or not hasattr(self, "fake_cls_head"):
            return
        path = os.path.join(self._fft_merged_model_dir(), "fake_cls_head.pth")
        if not os.path.exists(path):
            logger.warning(f"FFT fake classifier head checkpoint not found; keeping random initialization: {path}")
            return
        state_dict = torch.load(path, map_location="cpu")
        missing, unexpected = self.fake_cls_head.load_state_dict(state_dict, strict=False)
        logger.info(
            f"Loaded FFT fake classifier head from {path}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def init_optimizers(self):
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        optimizer_cls = self._optimizer_cls()

        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        self.G_opt = optimizer_cls(self.G_params, lr=self.config.lr_G, **self.config.opt_kwargs)
        if self._dmd_enabled():
            self.fake_params = list(filter(lambda p: p.requires_grad, self.fake_G.parameters()))
            if self._internal_gan_enabled():
                self.fake_params.extend(list(self.fake_cls_head.parameters()))
            self.fake_opt = optimizer_cls(self.fake_params, lr=self.config.lr_fake, **self.config.opt_kwargs)

        if self._external_d_enabled() and hasattr(self, "D"):
            self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
            self.D_opt = optimizer_cls(self.D_params, lr=self.config.lr_D, **self.config.opt_kwargs)

    def init_dataset(self):
        train_input_size = getattr(self.config, "input_size", 512)
        dataset = SDInpaintImageDataset(
            dataset_path=self.config.dataset_path,
            is_sdxl=True,
            input_size=train_input_size,
            use_blank_mask=getattr(self.config, "use_blank_mask", False),
            blank_mask_prob=getattr(self.config, "blank_mask_prob", 0.0),
            color_augmentation=getattr(self.config, "color_augmentation", False),
            flip_augmentation=getattr(self.config, "flip_augmentation", True),
            rotation_augmentation=getattr(self.config, "rotation_augmentation", False),
            rotation_prob=getattr(self.config, "rotation_prob", 0.5),
            random_mask_dilation=getattr(self.config, "random_mask_dilation", True),
            random_mask_erosion=getattr(self.config, "random_mask_erosion", True),
            return_minimal=True,
        )
        self.dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_num_workers,
            worker_init_fn=_rank_aware_worker_init_fn,
            pin_memory=getattr(self.config, "pin_memory", True),
            persistent_workers=bool(getattr(self.config, "persistent_workers", False))
            and self.config.dataloader_num_workers > 0,
            prefetch_factor=getattr(self.config, "prefetch_factor", 4) if self.config.dataloader_num_workers > 0 else None,
        )

        val_path = getattr(self.config, "validation_dataset_path", None)
        if val_path:
            val_input_size = getattr(self.config, "validation_input_size", train_input_size)
            val_max_samples = getattr(self.config, "validation_max_samples", -1)
            is_decoded_ober_dir = (
                os.path.isdir(val_path)
                and os.path.isdir(os.path.join(val_path, "image"))
                and os.path.isdir(os.path.join(val_path, "mask"))
                and os.path.isdir(os.path.join(val_path, "GT"))
            )
            if is_decoded_ober_dir:
                val_dataset = OBERDirectoryDataset(
                    root=val_path,
                    input_size=val_input_size,
                    max_samples=val_max_samples,
                    prompt=getattr(self.config, "validation_prompt", "remove the instance of object"),
                )
            else:
                val_dataset = SDInpaintImageDataset(
                    dataset_path=val_path,
                    is_sdxl=True,
                    input_size=val_input_size,
                    use_blank_mask=False,
                    blank_mask_prob=0.0,
                    color_augmentation=False,
                    flip_augmentation=False,
                    rotation_augmentation=False,
                    random_mask_dilation=False,
                    random_mask_erosion=False,
                )
                if val_max_samples is not None and int(val_max_samples) > 0:
                    val_dataset = torch.utils.data.Subset(val_dataset, range(min(int(val_max_samples), len(val_dataset))))
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=getattr(self.config, "validation_batch_size", 1),
                num_workers=getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers),
                worker_init_fn=_rank_aware_worker_init_fn,
                pin_memory=getattr(self.config, "pin_memory", True),
                persistent_workers=bool(getattr(self.config, "persistent_workers", False))
                and getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0,
                prefetch_factor=getattr(self.config, "prefetch_factor", 4)
                if getattr(self.config, "validation_num_workers", self.config.dataloader_num_workers) > 0
                else None,
            )
            logger.info(f"OBER validation dataset initialized with {len(val_dataset)} samples from {val_path}.")
        else:
            self.val_dataloader = None

    def _model_checkpoint_specs(self):
        specs = [("G", self.G, "state_dict.pth")]
        if self._dmd_enabled():
            specs.append(("fake_G", self.fake_G, "fake_state_dict.pth"))
        if self._dmd_enabled() and self._internal_gan_enabled():
            specs.append(("fake_cls_head", self.fake_cls_head, "fake_cls_head.pth"))
        return specs

    def _is_fsdp_training(self):
        return self.accelerator.distributed_type == DistributedType.FSDP

    def _fsdp_full_state_dict(self, model):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            return model.state_dict()

    def _save_fsdp_model_only_checkpoint(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        for role, model, filename in self._model_checkpoint_specs():
            state_dict = None
            if isinstance(model, FSDP):
                state_dict = self._fsdp_full_state_dict(model)
            elif self.accelerator.is_main_process:
                state_dict = self.unwrap_model(model).state_dict()

            if self.accelerator.is_main_process and state_dict is not None:
                torch.save(state_dict, os.path.join(save_path, filename))
                logger.info(f"Saved FSDP model-only {role} checkpoint to {filename}: tensors={len(state_dict)}")
                del state_dict
        self.accelerator.wait_for_everyone()

    def attach_accelerator_hooks(self):

        def save_model_hook(models, weights, output_dir):
            for role, model, filename in self._model_checkpoint_specs():
                if weights:
                    weights.pop(0)
                if not self.accelerator.is_main_process:
                    continue
                model = self.unwrap_model(model)
                if isinstance(model, UNet2DConditionModel):
                    state_dict = {}
                    for name, param in model.named_parameters():
                        if self._fft_enabled() or param.requires_grad:
                            state_dict[name] = param.detach().cpu().clone()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, filename))
                logger.info(f"Saved {role} checkpoint to {filename}: tensors={len(state_dict)}")

        def load_model_hook(models, input_dir):
            while models:
                models.pop()

            for role, model, filename in self._model_checkpoint_specs():
                path = os.path.join(input_dir, filename)
                if not os.path.exists(path) and filename == "fake_state_dict.pth":
                    logger.warning("fake_state_dict.pth missing; initializing fake_G from current generator state.")
                    path = os.path.join(input_dir, "state_dict.pth")
                if not os.path.exists(path) and filename == "fake_cls_head.pth":
                    logger.warning("fake_cls_head.pth missing; keeping randomly initialized fake classifier head.")
                    continue
                state_dict = torch.load(path, map_location="cpu")
                model = self.unwrap_model(model)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                del state_dict
                logger.info(
                    f"Loaded {role} from {filename}: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
                if missing:
                    logger.info(f"{role} missing keys sample: {missing[:10]}")
                if unexpected:
                    logger.info(f"{role} unexpected keys sample: {unexpected[:10]}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_all(self):
        logger.info("Wrapping DMD generator, fake model, optimizers and dataloaders")
        if self._load_model_only_on_resume() and not getattr(self, "_model_only_weights_loaded", False):
            self._load_model_only_weights(str(self.config.resume_from_checkpoint))
            self._model_only_weights_loaded = True
        if self._is_fsdp_training():
            model_attrs = ["G", "dataloader"]
            if self._dmd_enabled():
                model_attrs.insert(1, "fake_G")
                if self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
                    model_attrs.insert(2, "fake_cls_head")
            if self._external_d_enabled() and hasattr(self, "D"):
                model_attrs.insert(-1, "D")
            prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in model_attrs])
            for attr, obj in zip(model_attrs, prepared_objs):
                setattr(self, attr, obj)
            if hasattr(self, "teacher_G"):
                self.teacher_G.to(self.device).eval().requires_grad_(False)
            self._rebuild_optimizers_for_prepared_models()
            print_vram_state("After accelerator.prepare", logger=logger)
            return
        if not self._dmd_enabled():
            attrs = ["G", "G_opt", "dataloader"]
        else:
            attrs = ["G", "fake_G", "G_opt", "fake_opt", "dataloader"]
            if self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
                attrs.insert(2, "fake_cls_head")
        if self._external_d_enabled() and hasattr(self, "D") and hasattr(self, "D_opt"):
            attrs.extend(["D", "D_opt"])
        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        if hasattr(self, "teacher_G"):
            self.teacher_G.to(self.device).eval().requires_grad_(False)
        self._refresh_trainable_param_refs()
        print_vram_state("After accelerator.prepare", logger=logger)

    def _optimizer_cls(self):
        if self.config.optimizer_type == "adam":
            return torch.optim.AdamW
        if self.config.optimizer_type == "rmsprop":
            return torch.optim.RMSprop
        raise ValueError(f"Unsupported optimizer_type={self.config.optimizer_type}")

    @staticmethod
    def _prepared_sequence(prepared, expected_len):
        if expected_len == 1:
            return (prepared,)
        return prepared

    def _rebuild_optimizers_for_prepared_models(self):
        optimizer_cls = self._optimizer_cls()
        if hasattr(self.accelerator, "_optimizers"):
            self.accelerator._optimizers = []
        self._refresh_trainable_param_refs()
        self.G_opt = optimizer_cls(self.G_params, lr=self.config.lr_G, **self.config.opt_kwargs)
        opts = ["G_opt"]
        if self._dmd_enabled():
            self.fake_opt = optimizer_cls(self.fake_params, lr=self.config.lr_fake, **self.config.opt_kwargs)
            opts.append("fake_opt")
        if self._external_d_enabled() and hasattr(self, "D"):
            self.D_opt = optimizer_cls(self.D_params, lr=self.config.lr_D, **self.config.opt_kwargs)
            opts.append("D_opt")
        prepared_opts = self.accelerator.prepare(*[getattr(self, attr) for attr in opts])
        for attr, obj in zip(opts, self._prepared_sequence(prepared_opts, len(opts))):
            setattr(self, attr, obj)
        self._sync_optimizer_lrs_from_config()
        logger.info(
            "Rebuilt optimizers after FSDP model wrapping: "
            + ", ".join(f"{attr} groups={len(getattr(self, attr).param_groups)}" for attr in opts)
        )

    def _refresh_trainable_param_refs(self):
        if hasattr(self, "G"):
            self.G_params = [param for param in self.G.parameters() if param.requires_grad]
            logger.info(f"Refreshed G trainable parameter refs after prepare: {len(self.G_params)}")
        if hasattr(self, "fake_G"):
            self.fake_params = [param for param in self.fake_G.parameters() if param.requires_grad]
            if self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
                self.fake_params.extend(list(self.fake_cls_head.parameters()))
            logger.info(f"Refreshed fake trainable parameter refs after prepare: {len(self.fake_params)}")
        if hasattr(self, "D"):
            self.D_params = [param for param in self.D.parameters() if param.requires_grad]
            logger.info(f"Refreshed D trainable parameter refs after prepare: {len(self.D_params)}")

    def on_training_start(self):
        if self._load_model_only_on_resume():
            self._resume_model_only()
            self._sync_optimizer_lrs_from_config()
            self.debug_saved_steps = 0
            return
        try:
            super().on_training_start()
        except ValueError as exc:
            if not self._is_resume_optimizer_mismatch(exc):
                raise
            self._resume_with_fresh_optimizers_after_model_load(exc)
        self._sync_optimizer_lrs_from_config()
        self.debug_saved_steps = 0

    def _load_model_only_on_resume(self):
        return bool(getattr(self.config, "resume_model_only", False)) and bool(
            getattr(self.config, "resume_from_checkpoint", None)
        )

    def _create_ema_handler(self, ema_resume_pth=None):
        logger.info(f"Creating EMA handler, Use EMA = {self.config.use_ema}, EMA decay = {self.config.ema_decay}")
        self.ema_handler = EMAModel(
            self.unwrap_model(self.G),
            decay=self.config.ema_decay,
            use_ema=self.config.use_ema,
            ema_resume_pth=ema_resume_pth,
            verbose=self.accelerator.is_local_main_process,
        )

    def _resume_model_only(self):
        path = str(self.config.resume_from_checkpoint)
        logger.info(f"Resuming model weights only from checkpoint {path}")
        self._create_ema_handler(ema_resume_pth=None)
        if not getattr(self, "_model_only_weights_loaded", False):
            self._load_model_only_weights(path)
            self._model_only_weights_loaded = True
        self._setup_model_only_resume_state(path)

    def _load_model_only_weights(self, path):
        specs = [("G", self.G, "state_dict.pth")]
        if self._dmd_enabled() and hasattr(self, "fake_G"):
            specs.append(("fake_G", self.fake_G, "fake_state_dict.pth"))
        if self._dmd_enabled() and self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
            specs.append(("fake_cls_head", self.fake_cls_head, "fake_cls_head.pth"))

        for role, model, filename in specs:
            state_path = os.path.join(path, filename)
            if not os.path.exists(state_path):
                if filename == "fake_cls_head.pth":
                    logger.warning("fake_cls_head.pth missing; keeping current fake classifier head.")
                    continue
                raise FileNotFoundError(f"Required {role} checkpoint is missing for model-only resume: {state_path}")
            state_dict = torch.load(state_path, map_location="cpu")
            target = self.unwrap_model(model)
            missing, unexpected = target.load_state_dict(state_dict, strict=False)
            del state_dict
            logger.info(
                f"Loaded {role} model-only from {filename}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def _setup_model_only_resume_state(self, path):
        ckpt_name = os.path.basename(path.rstrip("/"))
        try:
            global_step = int(ckpt_name.split("-")[1])
        except Exception:
            global_step = 0
        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_optimizer_lr(self, optimizer, lr, name):
        if optimizer is None:
            return
        lr = float(lr)
        for group in optimizer.param_groups:
            old_lr = float(group.get("lr", lr))
            group["lr"] = lr
            if "initial_lr" in group:
                group["initial_lr"] = lr
            if old_lr != lr:
                logger.info(f"Reset {name} optimizer lr after resume: {old_lr} -> {lr}")

    def _sync_optimizer_lrs_from_config(self):
        self._set_optimizer_lr(getattr(self, "G_opt", None), self.config.lr_G, "G")
        if hasattr(self, "fake_opt"):
            self._set_optimizer_lr(self.fake_opt, self.config.lr_fake, "fake")
        if hasattr(self, "D_opt"):
            self._set_optimizer_lr(self.D_opt, self.config.lr_D, "D")

    def _is_resume_optimizer_mismatch(self, exc):
        return "loaded state dict contains a parameter group that doesn't match the size of optimizer's group" in str(exc)

    def _resume_with_fresh_optimizers_after_model_load(self, exc):
        allow_partial = bool(getattr(self.config, "resume_allow_partial_optimizer", True))
        if not allow_partial:
            raise exc
        path = self.config.resume_from_checkpoint
        ckpt_name = os.path.basename(path)
        logger.warning(
            "Optimizer state in %s is incompatible with the current DMD optimizer layout. "
            "Keeping loaded model weights and continuing with freshly initialized optimizers. "
            "Original error: %s",
            path,
            exc,
        )

        self._create_ema_handler(ema_resume_pth=None)
        global_step = int(ckpt_name.split("-")[1])
        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.max_train_steps),
            initial=global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )
        self._rebuild_optimizers_after_partial_resume()

    def _rebuild_optimizers_after_partial_resume(self):
        optimizer_cls = self._optimizer_cls()

        if hasattr(self.accelerator, "_optimizers"):
            self.accelerator._optimizers = []

        self.G_params = [p for p in self.G.parameters() if p.requires_grad]
        self.G_opt = optimizer_cls(self.G_params, lr=self.config.lr_G, **self.config.opt_kwargs)
        if not self._dmd_enabled():
            self.G_opt = self.accelerator.prepare(self.G_opt)
        else:
            self.fake_params = [p for p in self.fake_G.parameters() if p.requires_grad]
            if self._internal_gan_enabled():
                self.fake_params.extend([p for p in self.fake_cls_head.parameters() if p.requires_grad])
            self.fake_opt = optimizer_cls(self.fake_params, lr=self.config.lr_fake, **self.config.opt_kwargs)
            self.G_opt, self.fake_opt = self.accelerator.prepare(self.G_opt, self.fake_opt)

        if self._external_d_enabled() and hasattr(self, "D"):
            self.D_params = [p for p in self.D.parameters() if p.requires_grad]
            self.D_opt = optimizer_cls(self.D_params, lr=self.config.lr_D, **self.config.opt_kwargs)
            self.D_opt = self.accelerator.prepare(self.D_opt)

        self._refresh_trainable_param_refs()
        logger.info("Rebuilt optimizers after partial resume; optimizer states start fresh.")

    def _internal_gan_enabled(self):
        return self._dmd_enabled() and bool(getattr(self.config, "use_gan", False))

    def _external_d_enabled(self):
        return bool(getattr(self.config, "use_external_D", False))

    def _clip_grad_norm(self, model, params, max_norm):
        params = [param for param in params if param is not None]
        if not params:
            return torch.tensor(0.0, device=self.device)

        models = list(model) if isinstance(model, (list, tuple)) else [model]
        param_ids = {id(param) for param in params}
        covered_param_ids = set()
        norm_tensors = []
        for candidate in models:
            if not isinstance(candidate, FSDP):
                continue
            model_params = list(candidate.parameters())
            model_param_ids = {id(param) for param in model_params}
            if not (param_ids & model_param_ids):
                continue
            norm_tensors.append(candidate.clip_grad_norm_(max_norm).to(device=self.device))
            covered_param_ids.update(model_param_ids)

        extra_params = [param for param in params if id(param) not in covered_param_ids]
        if extra_params:
            extra_norm = torch.nn.utils.clip_grad_norm_(extra_params, max_norm)
            if not isinstance(extra_norm, torch.Tensor):
                extra_norm = torch.tensor(float(extra_norm), device=self.device)
            norm_tensors.append(extra_norm.to(device=self.device))

        if not norm_tensors:
            return torch.tensor(0.0, device=self.device)
        norm = norm_tensors[0]
        for next_norm in norm_tensors[1:]:
            norm = torch.maximum(norm, next_norm.to(dtype=norm.dtype))
        return norm

    @contextmanager
    def _unet_forward_context(self):
        if bool(getattr(self.config, "unet_forward_autocast", False)):
            device_type = "cuda" if self.device.type == "cuda" else self.device.type
            with torch.autocast(device_type=device_type, dtype=self.weight_dtype):
                yield
        else:
            yield

    @contextmanager
    def _disable_unet_forward_autocast_temporarily(self):
        old_value = getattr(self.config, "unet_forward_autocast", False)
        self.config.unet_forward_autocast = False
        try:
            yield
        finally:
            self.config.unet_forward_autocast = old_value

    @contextmanager
    def _temporarily_freeze_fake_score_model(self):
        modules = [self.fake_G]
        if self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
            modules.append(self.fake_cls_head)
        params = []
        for module in modules:
            params.extend(list(module.parameters()))
        states = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(False)
            yield
        finally:
            for param, state in zip(params, states):
                param.requires_grad_(state)

    def _repeat_condition_to_batch(self, tensor, batch_size, name):
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] <= 0 or batch_size % tensor.shape[0] != 0:
            raise ValueError(
                f"{name} batch size {tensor.shape[0]} cannot be repeated to match latent batch size {batch_size}."
            )
        repeats = [batch_size // tensor.shape[0]] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeats)

    def _unet_pred(self, model, latents, timesteps):
        model_dtype = next(model.parameters()).dtype
        batch_size = latents.shape[0]
        if timesteps.ndim > 0:
            timesteps = self._repeat_condition_to_batch(timesteps, batch_size, "timesteps")
        mask_latent = self._repeat_condition_to_batch(
            self.batch_inputs.mask_latent, batch_size, "mask_latent"
        ).to(dtype=model_dtype)
        masked_image_latents = self._repeat_condition_to_batch(
            self.batch_inputs.masked_image_latents, batch_size, "masked_image_latents"
        ).to(dtype=model_dtype)
        prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["prompt_embeds"], batch_size, "prompt_embeds"
        ).to(dtype=model_dtype)
        pooled_prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["pooled_prompt_embeds"], batch_size, "pooled_prompt_embeds"
        ).to(dtype=model_dtype)
        add_time_ids = self._repeat_condition_to_batch(
            self.batch_inputs.add_time_ids, batch_size, "add_time_ids"
        ).to(dtype=model_dtype)
        model_input = self.scheduler.scale_model_input(latents, timesteps)
        model_input = model_input.to(dtype=model_dtype)
        model_input = torch.cat(
            [
                model_input,
                mask_latent,
                masked_image_latents,
            ],
            dim=1,
        )
        with self._unet_forward_context():
            return model(
                model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample

    def _unet_pred_cfg(self, model, latents, timesteps, guidance_scale):
        guidance_scale = float(guidance_scale)
        if guidance_scale <= 1.0:
            return self._unet_pred(model, latents, timesteps)

        model_dtype = next(model.parameters()).dtype
        batch_size = latents.shape[0]
        if timesteps.ndim > 0:
            timesteps = self._repeat_condition_to_batch(timesteps, batch_size, "timesteps")
        mask_latent = self._repeat_condition_to_batch(
            self.batch_inputs.mask_latent, batch_size, "mask_latent"
        ).to(dtype=model_dtype)
        masked_image_latents = self._repeat_condition_to_batch(
            self.batch_inputs.masked_image_latents, batch_size, "masked_image_latents"
        ).to(dtype=model_dtype)
        model_input = self.scheduler.scale_model_input(latents, timesteps)
        model_input = model_input.to(dtype=model_dtype)
        model_input = torch.cat(
            [
                model_input,
                mask_latent,
                masked_image_latents,
            ],
            dim=1,
        )
        model_input = torch.cat([model_input, model_input], dim=0)
        timesteps = torch.cat([timesteps, timesteps], dim=0)
        prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["prompt_embeds"], batch_size, "prompt_embeds"
        ).to(dtype=model_dtype)
        pooled_prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["pooled_prompt_embeds"], batch_size, "pooled_prompt_embeds"
        ).to(dtype=model_dtype)
        uncond_prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.uncond_txt["prompt_embeds"], batch_size, "uncond_prompt_embeds"
        ).to(
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        uncond_pooled_prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.uncond_txt["pooled_prompt_embeds"], batch_size, "uncond_pooled_prompt_embeds"
        ).to(
            device=pooled_prompt_embeds.device,
            dtype=pooled_prompt_embeds.dtype,
        )
        add_time_ids = self._repeat_condition_to_batch(
            self.batch_inputs.add_time_ids, batch_size, "add_time_ids"
        ).to(dtype=model_dtype)
        encoder_hidden_states = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
        added_cond_kwargs = {
            "text_embeds": torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds], dim=0),
            "time_ids": torch.cat([add_time_ids, add_time_ids], dim=0),
        }

        with self._unet_forward_context():
            noise_pred = model(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    def _unet_pred_fp32_probe(self, model, latents, timesteps):
        batch_size = latents.shape[0]
        if timesteps.ndim > 0:
            timesteps = self._repeat_condition_to_batch(timesteps, batch_size, "timesteps")
        mask_latent = self._repeat_condition_to_batch(
            self.batch_inputs.mask_latent, batch_size, "mask_latent"
        ).float()
        masked_image_latents = self._repeat_condition_to_batch(
            self.batch_inputs.masked_image_latents, batch_size, "masked_image_latents"
        ).float()
        prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["prompt_embeds"], batch_size, "prompt_embeds"
        ).float()
        pooled_prompt_embeds = self._repeat_condition_to_batch(
            self.batch_inputs.c_txt["pooled_prompt_embeds"], batch_size, "pooled_prompt_embeds"
        ).float()
        add_time_ids = self._repeat_condition_to_batch(
            self.batch_inputs.add_time_ids, batch_size, "add_time_ids"
        ).float()
        model_input = self.scheduler.scale_model_input(latents.float(), timesteps)
        model_input = torch.cat(
            [
                model_input,
                mask_latent,
                masked_image_latents,
            ],
            dim=1,
        )
        with self._disable_unet_forward_autocast_temporarily():
            return model(
                model_input,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                },
            ).sample

    def _cfg_scale(self, primary_name, alias_name, default):
        value = getattr(self.config, primary_name, None)
        if value is None:
            value = getattr(self.config, alias_name, default)
        return float(value)

    def _pred_x0(self, model_output, sample, timesteps):
        prediction_type = self.scheduler.config.prediction_type
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t = 1.0 - alpha_prod_t

        if prediction_type == "epsilon":
            pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        return pred_x0

    def _add_noise_at(self, clean_latents, noise, timesteps):
        return self.scheduler.add_noise(clean_latents, noise, timesteps)

    def _masked_l1(self, pred, target, mask):
        loss = (pred.float() - target.float()).abs()
        mask = mask.to(device=loss.device, dtype=loss.dtype)
        if mask.shape[1] == 1 and loss.shape[1] != 1:
            mask = mask.repeat(1, loss.shape[1], 1, 1)
        return (loss * mask).sum() / (mask.sum() + 1e-8)

    @staticmethod
    def _masked_psnr(pred, gt, mask):
        pred_01 = ((pred.float() + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((gt.float() + 1.0) / 2.0).clamp(0, 1)
        mask = (mask.float() > 0.5).float().to(device=pred_01.device)
        if mask.shape[1] == 1 and pred_01.shape[1] != 1:
            mask = mask.repeat(1, pred_01.shape[1], 1, 1)
        mse = ((pred_01 - gt_01) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        mse_val = float(mse.detach().cpu())
        return -10.0 * np.log10(max(mse_val, 1e-12))

    @staticmethod
    def _full_psnr(pred, gt):
        pred_01 = ((pred.float() + 1.0) / 2.0).clamp(0, 1)
        gt_01 = ((gt.float() + 1.0) / 2.0).clamp(0, 1)
        mse = ((pred_01 - gt_01) ** 2).mean()
        mse_val = float(mse.detach().cpu())
        return -10.0 * np.log10(max(mse_val, 1e-12))

    @staticmethod
    def _numeric_tensor_stats(prefix, tensor):
        if not isinstance(tensor, torch.Tensor):
            return {}
        data = tensor.detach().float()
        if data.numel() == 0:
            zero = torch.tensor(0.0, device=tensor.device)
            return {
                f"{prefix}_abs_mean": zero,
                f"{prefix}_abs_max": zero,
                f"{prefix}_nonfinite_ratio": zero,
            }
        finite = torch.isfinite(data)
        safe = torch.where(finite, data, torch.zeros_like(data))
        return {
            f"{prefix}_abs_mean": safe.abs().mean(),
            f"{prefix}_abs_max": safe.abs().amax(),
            f"{prefix}_nonfinite_ratio": 1.0 - finite.float().mean(),
        }

    @staticmethod
    def _zero_like_loss(reference):
        return torch.tensor(0.0, device=reference.device, dtype=reference.dtype)

    def _model_param_stats(self, prefix, model, max_tensors=None):
        if max_tensors is None:
            max_tensors = int(getattr(self.config, "debug_param_stat_tensors", 16))
        abs_sum = torch.tensor(0.0, device=self.device)
        max_abs = torch.tensor(0.0, device=self.device)
        nonfinite_sum = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for idx, param in enumerate(self.unwrap_model(model).parameters()):
                if max_tensors >= 0 and idx >= max_tensors:
                    break
                data = param.detach().float()
                if data.numel() == 0:
                    continue
                finite = torch.isfinite(data)
                safe = torch.where(finite, data, torch.zeros_like(data))
                abs_sum = abs_sum + safe.abs().mean().to(self.device)
                max_abs = torch.maximum(max_abs, safe.abs().amax().to(self.device))
                nonfinite_sum = nonfinite_sum + (1.0 - finite.float().mean()).to(self.device)
                count = count + 1.0
        denom = count.clamp_min(1.0)
        return {
            f"{prefix}_param_abs_mean": abs_sum / denom,
            f"{prefix}_param_abs_max": max_abs,
            f"{prefix}_param_nonfinite_ratio": nonfinite_sum / denom,
            f"{prefix}_param_stat_count": count,
        }

    @staticmethod
    def _optimizer_params(optimizer):
        if optimizer is None:
            return []
        params = []
        for group in optimizer.param_groups:
            params.extend(group.get("params", []))
        return [param for param in params if param is not None]

    def _lpips_trace_enabled(self):
        return bool(getattr(self.config, "debug_lpips_trace_mode", False))

    def _lpips_trace_rank_dir(self, step):
        root = getattr(
            self.config,
            "debug_lpips_trace_dir",
            os.path.join(self.config.output_dir, "debug_lpips_trace"),
        )
        rank = int(getattr(self.accelerator, "process_index", 0))
        path = os.path.join(root, f"step_{int(step):06d}", f"rank_{rank:03d}")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _debug_scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, (int, float, bool)):
            return float(value)
        return value

    @staticmethod
    def _tensor_debug_stats(tensor):
        if not isinstance(tensor, torch.Tensor):
            return {}
        data = tensor.detach().float()
        finite = torch.isfinite(data)
        safe = torch.where(finite, data, torch.zeros_like(data))
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "abs_mean": float(safe.abs().mean().cpu().item()) if safe.numel() else 0.0,
            "abs_max": float(safe.abs().amax().cpu().item()) if safe.numel() else 0.0,
            "min": float(safe.amin().cpu().item()) if safe.numel() else 0.0,
            "max": float(safe.amax().cpu().item()) if safe.numel() else 0.0,
            "nonfinite_ratio": float((1.0 - finite.float().mean()).cpu().item()) if safe.numel() else 0.0,
        }

    def _debug_save_image_tensor(self, tensor, path, mask=False):
        image = tensor.detach().float().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if not mask:
            image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)
        arr = (image * 255.0).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(arr).save(path)

    @staticmethod
    def _debug_save_grayscale_tensor(tensor, path):
        image = tensor.detach().float().cpu().squeeze().clamp(0, 1)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale tensor, got shape={tuple(image.shape)}")
        arr = (image * 255.0).round().to(torch.uint8).contiguous().numpy()
        Image.fromarray(arr).save(path)

    @staticmethod
    def _debug_turbo_colormap(tensor):
        x = tensor.detach().float().clamp(0, 1)
        coefficients = (
            (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
            (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
            (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
        )
        channels = []
        for channel_coefficients in coefficients:
            channel = torch.full_like(x, channel_coefficients[-1])
            for coefficient in reversed(channel_coefficients[:-1]):
                channel = channel * x + coefficient
            channels.append(channel.clamp(0, 1))
        return torch.stack(channels, dim=0)

    def _debug_param_snapshot(self, params, max_tensors=None):
        if max_tensors is None:
            max_tensors = int(getattr(self.config, "debug_lpips_trace_param_tensors", 8))
        stats = []
        with torch.no_grad():
            for idx, param in enumerate(params):
                if idx >= max_tensors:
                    break
                data = param.detach().float()
                finite = torch.isfinite(data)
                safe = torch.where(finite, data, torch.zeros_like(data))
                flat = safe.flatten()
                stats.append(
                    {
                        "idx": idx,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "requires_grad": bool(param.requires_grad),
                        "abs_mean": float(safe.abs().mean().cpu().item()) if safe.numel() else 0.0,
                        "abs_max": float(safe.abs().amax().cpu().item()) if safe.numel() else 0.0,
                        "checksum_1024": float(flat[: min(1024, flat.numel())].sum().cpu().item())
                        if flat.numel()
                        else 0.0,
                        "nonfinite_ratio": float((1.0 - finite.float().mean()).cpu().item())
                        if safe.numel()
                        else 0.0,
                    }
                )
        return stats

    @staticmethod
    def _debug_param_delta(before, after):
        deltas = []
        for before_item, after_item in zip(before, after):
            deltas.append(
                {
                    "idx": before_item["idx"],
                    "checksum_delta_abs": abs(after_item["checksum_1024"] - before_item["checksum_1024"]),
                    "abs_mean_delta_abs": abs(after_item["abs_mean"] - before_item["abs_mean"]),
                    "abs_max_delta_abs": abs(after_item["abs_max"] - before_item["abs_max"]),
                    "before": before_item,
                    "after": after_item,
                }
            )
        return deltas

    def _debug_delta_summary(self, deltas, prefix):
        if not deltas:
            zero = torch.tensor(0.0, device=self.device)
            return {
                f"{prefix}_param_delta_checksum_abs_max": zero,
                f"{prefix}_param_delta_abs_mean_max": zero,
                f"{prefix}_param_delta_abs_max_max": zero,
                f"{prefix}_param_update_nonzero": zero,
            }
        checksum = max(item["checksum_delta_abs"] for item in deltas)
        abs_mean = max(item["abs_mean_delta_abs"] for item in deltas)
        abs_max = max(item["abs_max_delta_abs"] for item in deltas)
        nonzero = 1.0 if checksum > 0.0 or abs_mean > 0.0 or abs_max > 0.0 else 0.0
        return {
            f"{prefix}_param_delta_checksum_abs_max": torch.tensor(float(checksum), device=self.device),
            f"{prefix}_param_delta_abs_mean_max": torch.tensor(float(abs_mean), device=self.device),
            f"{prefix}_param_delta_abs_max_max": torch.tensor(float(abs_max), device=self.device),
            f"{prefix}_param_update_nonzero": torch.tensor(nonzero, device=self.device),
        }

    def _param_update_probe(self, params, max_tensors=None, max_elements=None):
        if max_tensors is None:
            max_tensors = int(getattr(self.config, "debug_optimizer_delta_tensors", -1))
        if max_elements is None:
            max_elements = int(getattr(self.config, "debug_optimizer_delta_elements", 4096))
        probes = []
        with torch.no_grad():
            for idx, param in enumerate(params):
                if max_tensors >= 0 and len(probes) >= max_tensors:
                    break
                data = param.detach()
                if data.numel() == 0:
                    continue
                flat = data.float().flatten()
                probe_len = min(max_elements, flat.numel())
                probes.append(
                    {
                        "idx": idx,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "values": flat[:probe_len].cpu().clone(),
                    }
                )
        return probes

    def _param_update_probe_delta_summary(self, before, after, prefix):
        if not before or not after:
            zero = torch.tensor(0.0, device=self.device)
            return {
                f"{prefix}_probe_delta_abs_max": zero,
                f"{prefix}_probe_delta_abs_mean_max": zero,
                f"{prefix}_probe_update_nonzero": zero,
                f"{prefix}_probe_tensor_count": zero,
            }
        by_idx = {item["idx"]: item for item in after}
        max_delta = 0.0
        max_mean_delta = 0.0
        count = 0
        for before_item in before:
            after_item = by_idx.get(before_item["idx"])
            if after_item is None:
                continue
            before_values = before_item["values"]
            after_values = after_item["values"]
            probe_len = min(before_values.numel(), after_values.numel())
            if probe_len == 0:
                continue
            delta = (after_values[:probe_len] - before_values[:probe_len]).abs()
            max_delta = max(max_delta, float(delta.max().item()))
            max_mean_delta = max(max_mean_delta, float(delta.mean().item()))
            count += 1
        nonzero = 1.0 if max_delta > 0.0 or max_mean_delta > 0.0 else 0.0
        return {
            f"{prefix}_probe_delta_abs_max": torch.tensor(max_delta, device=self.device),
            f"{prefix}_probe_delta_abs_mean_max": torch.tensor(max_mean_delta, device=self.device),
            f"{prefix}_probe_update_nonzero": torch.tensor(nonzero, device=self.device),
            f"{prefix}_probe_tensor_count": torch.tensor(float(count), device=self.device),
        }

    def _debug_grad_stats(self, params):
        total_sq = torch.tensor(0.0, device=self.device)
        max_abs = torch.tensor(0.0, device=self.device)
        grad_count = 0
        none_count = 0
        nonfinite_sum = 0.0
        examples = []
        for idx, param in enumerate(params):
            grad = param.grad
            if grad is None:
                none_count += 1
                continue
            grad_count += 1
            data = grad.detach().float()
            finite = torch.isfinite(data)
            safe = torch.where(finite, data, torch.zeros_like(data))
            total_sq = total_sq + safe.pow(2).sum().to(self.device)
            if safe.numel():
                max_abs = torch.maximum(max_abs, safe.abs().amax().to(self.device))
            nonfinite_sum += float((1.0 - finite.float().mean()).cpu().item()) if safe.numel() else 0.0
            if len(examples) < int(getattr(self.config, "debug_lpips_trace_grad_tensors", 8)):
                examples.append(
                    {
                        "idx": idx,
                        "shape": list(param.shape),
                        "param_dtype": str(param.dtype),
                        "grad_dtype": str(grad.dtype),
                        "grad_abs_mean": float(safe.abs().mean().cpu().item()) if safe.numel() else 0.0,
                        "grad_abs_max": float(safe.abs().amax().cpu().item()) if safe.numel() else 0.0,
                        "grad_nonfinite_ratio": float((1.0 - finite.float().mean()).cpu().item())
                        if safe.numel()
                        else 0.0,
                    }
                )
        return {
            "grad_norm": float(total_sq.sqrt().detach().cpu().item()),
            "grad_abs_max": float(max_abs.detach().cpu().item()),
            "grad_param_count": grad_count,
            "grad_none_count": none_count,
            "grad_nonfinite_ratio_mean": nonfinite_sum / max(grad_count, 1),
            "examples": examples,
        }

    def _write_lpips_trace(
        self,
        step,
        loss_items,
        lpips_per_sample,
        params_before,
        params_after,
        grad_before_clip,
        grad_after_clip,
        clip_return,
    ):
        if not self._lpips_trace_enabled():
            return
        rank_dir = self._lpips_trace_rank_dir(step)
        bs = int(self.G_pred.shape[0]) if hasattr(self, "G_pred") else int(self.batch_inputs.gt.shape[0])
        sample_records = []
        lpips_values = []
        if isinstance(lpips_per_sample, torch.Tensor):
            lpips_values = [float(x) for x in lpips_per_sample.detach().float().cpu().view(-1).tolist()]

        for idx in range(bs):
            sample_dir = os.path.join(rank_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            self._debug_save_image_tensor(self.batch_inputs.input_img[idx], os.path.join(sample_dir, "input.png"))
            self._debug_save_image_tensor(self.batch_inputs.gt[idx], os.path.join(sample_dir, "gt_lpips_target.png"))
            self._debug_save_image_tensor(self.G_pred[idx], os.path.join(sample_dir, "pred_lpips_input.png"))
            self._debug_save_image_tensor(
                self.batch_inputs.object_mask[idx],
                os.path.join(sample_dir, "object_mask.png"),
                mask=True,
            )
            self._debug_save_image_tensor(
                self.batch_inputs.object_effect_mask[idx],
                os.path.join(sample_dir, "effect_mask.png"),
                mask=True,
            )
            abs_err = (self.G_pred[idx].detach().float() - self.batch_inputs.gt[idx].detach().float()).abs()
            abs_err = abs_err / (abs_err.amax() + 1e-8)
            self._debug_save_image_tensor(abs_err * 2.0 - 1.0, os.path.join(sample_dir, "abs_error_norm.png"))
            sample_records.append(
                {
                    "sample_index": idx,
                    "prompt": self.batch_inputs.prompt[idx] if idx < len(self.batch_inputs.prompt) else "",
                    "lpips": lpips_values[idx] if idx < len(lpips_values) else None,
                    "input_stats": self._tensor_debug_stats(self.batch_inputs.input_img[idx]),
                    "gt_lpips_target_stats": self._tensor_debug_stats(self.batch_inputs.gt[idx]),
                    "pred_lpips_input_stats": self._tensor_debug_stats(self.G_pred[idx]),
                    "mask_stats": self._tensor_debug_stats(self.batch_inputs.object_mask[idx]),
                    "effect_mask_stats": self._tensor_debug_stats(self.batch_inputs.object_effect_mask[idx]),
                    "z_lq_stats": self._tensor_debug_stats(self.batch_inputs.z_lq[idx]),
                    "z_gt_stats": self._tensor_debug_stats(self.batch_inputs.z_gt[idx]),
                }
            )

        opt_lrs = []
        if hasattr(self, "G_opt"):
            opt_lrs = [float(group.get("lr", 0.0)) for group in self.G_opt.param_groups]
        payload = {
            "step": int(step),
            "rank": int(getattr(self.accelerator, "process_index", 0)),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": int(getattr(self.accelerator, "num_processes", 1)),
            "batch_size_local": bs,
            "batch_count": int(getattr(self, "batch_count", -1)),
            "optimizer_lrs": opt_lrs,
            "G_param_count": len(getattr(self, "G_params", [])),
            "losses": {key: self._debug_scalar(value) for key, value in loss_items.items()},
            "lpips_per_sample": lpips_values,
            "timesteps": {
                "generator_noise_timestep": self._debug_scalar(
                    getattr(self, "last_generator_noise_timestep", torch.tensor(0.0))
                ),
                "generator_model_timestep": self._debug_scalar(
                    getattr(self, "last_generator_model_timestep", torch.tensor(0.0))
                ),
                "diffusion_noise_timestep": self._debug_scalar(
                    getattr(self, "last_diffusion_noise_timestep", torch.tensor(0.0))
                ),
                "diffusion_model_timestep": self._debug_scalar(
                    getattr(self, "last_diffusion_model_timestep", torch.tensor(0.0))
                ),
            },
            "tensor_stats": {
                "prompt_embeds": self._tensor_debug_stats(self.batch_inputs.c_txt["prompt_embeds"]),
                "pooled_prompt_embeds": self._tensor_debug_stats(self.batch_inputs.c_txt["pooled_prompt_embeds"]),
                "x_fake_latents": self._tensor_debug_stats(getattr(self, "G_pred_latents", None)),
                "x_pred": self._tensor_debug_stats(getattr(self, "G_pred", None)),
                "gt": self._tensor_debug_stats(self.batch_inputs.gt),
                "input": self._tensor_debug_stats(self.batch_inputs.input_img),
            },
            "grad_before_clip": grad_before_clip,
            "grad_after_clip": grad_after_clip,
            "clip_return_norm": self._debug_scalar(clip_return),
            "params_before": params_before,
            "params_after": params_after,
            "param_deltas": self._debug_param_delta(params_before, params_after),
            "samples": sample_records,
        }
        payload["param_update_nonzero"] = any(
            item["checksum_delta_abs"] > 0.0
            or item["abs_mean_delta_abs"] > 0.0
            or item["abs_max_delta_abs"] > 0.0
            for item in payload["param_deltas"]
        )
        payload["grad_nonzero_before_clip"] = grad_before_clip.get("grad_norm", 0.0) > 0.0
        payload["grad_nonzero_after_clip"] = grad_after_clip.get("grad_norm", 0.0) > 0.0
        with open(os.path.join(rank_dir, "trace.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _lpips_gt_loss(self, pred, target):
        if not bool(getattr(self.config, "local_lpips", False)):
            return self.net_lpips(pred, target).mean()

        mask = self.batch_inputs.object_effect_mask.to(device=pred.device)
        if mask.shape[-2:] != pred.shape[-2:]:
            mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")
        mask = mask[:, :1].float() > 0.5
        padding = max(0, int(getattr(self.config, "local_lpips_padding_pixels", 0)))

        losses = []
        _, _, height, width = pred.shape
        for idx in range(pred.shape[0]):
            ys, xs = torch.where(mask[idx, 0])
            if ys.numel() == 0 or xs.numel() == 0:
                pred_crop = pred[idx : idx + 1]
                target_crop = target[idx : idx + 1]
            else:
                y0 = max(0, int(ys.min().item()) - padding)
                y1 = min(height, int(ys.max().item()) + padding + 1)
                x0 = max(0, int(xs.min().item()) - padding)
                x1 = min(width, int(xs.max().item()) + padding + 1)
                pred_crop = pred[idx : idx + 1, :, y0:y1, x0:x1]
                target_crop = target[idx : idx + 1, :, y0:y1, x0:x1]
            losses.append(self.net_lpips(pred_crop, target_crop).mean())
        return torch.stack(losses).mean()

    def _decode_latents(self, latents):
        z = latents.to(self.weight_dtype) / self.vae.config.scaling_factor
        if torch.is_grad_enabled() and z.requires_grad and getattr(self.config, "vae_decode_checkpointing", True):
            return checkpoint(lambda x: self.vae.decode(x).sample, z, use_reentrant=False).float()
        return self.vae.decode(z).sample.float()

    @staticmethod
    def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
        x = ((x + 1.0) / 2.0).clamp(0, 1)
        x = (x * 255.0).round().to(torch.uint8)
        return x.permute(1, 2, 0).contiguous().cpu().numpy()

    @staticmethod
    def _uint8_image_to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        t = t * 2.0 - 1.0
        return t.to(device=device, dtype=dtype)

    def _validation_attention_guided_fusion(self, x_pred, z_pred):
        if not bool(getattr(self.config, "apply_attention_guided_fusion_in_validation", False)):
            return x_pred, z_pred
        if len(self.cross_attention_scores) == 0:
            return x_pred, z_pred

        mask = self.batch_inputs.mask_latent.to(dtype=z_pred.dtype)
        fuse_index = getattr(self.config, "fuse_index", 5)
        _, layer_attn = next(iter(self.cross_attention_scores.items()))
        attn_map = resize_attn_map_divide2(layer_attn, mask, fuse_index).mean(dim=1, keepdim=True)

        z_fused = (1.0 - attn_map.to(dtype=z_pred.dtype)) * self.batch_inputs.masked_image_latents.to(
            dtype=z_pred.dtype
        ) + attn_map.to(dtype=z_pred.dtype) * z_pred
        x_fused = self._decode_latents(z_fused)

        fused_images = []
        for idx in range(x_fused.shape[0]):
            ori_np = self._to_uint8_image(self.batch_inputs.input_img[idx])
            pred_np = self._to_uint8_image(x_fused[idx])
            attn_np = (attn_map[idx].mean(dim=0).float().detach().cpu().numpy() * 255.0).astype(np.uint8)
            fused_np = attention_guided_fusion(ori_np, pred_np, attn_np)
            fused_images.append(self._uint8_image_to_tensor(fused_np, x_fused.device, x_fused.dtype))
        return torch.stack(fused_images, dim=0), z_fused

    def _select_generator_timestep(self, batch_size):
        configured = getattr(self.config, "generator_timestep", None)
        if configured is not None:
            return torch.full((batch_size,), int(configured), dtype=torch.long, device=self.device)
        timesteps = self._set_inference_timesteps(int(getattr(self.config, "student_steps", 1)))
        return torch.full((batch_size,), int(timesteps[0]), dtype=torch.long, device=self.device)

    def _select_generator_timesteps(self, batch_size):
        noise_timesteps = self._set_inference_timesteps(int(getattr(self.config, "student_steps", 1)))
        noise_timesteps = torch.full((batch_size,), int(noise_timesteps[0]), dtype=torch.long, device=self.device)
        if str(getattr(self.config, "scheduler_timestep_spacing", "trailing")).lower() != "fixed":
            return noise_timesteps, noise_timesteps
        configured_model_timestep = getattr(self.config, "generator_timestep", None)
        if configured_model_timestep is None:
            return noise_timesteps, noise_timesteps
        if int(getattr(self.config, "student_steps", 1)) != 1:
            raise ValueError("generator_timestep is only supported for 1-step generator inference.")
        model_timesteps = torch.full(
            (batch_size,),
            int(configured_model_timestep),
            dtype=torch.long,
            device=self.device,
        )
        return noise_timesteps, model_timesteps

    def _copy_latent_background(self, latents):
        if not bool(getattr(self.config, "latent_background_copy", False)):
            return latents
        edit_mask = self.batch_inputs.mask_latent.to(dtype=latents.dtype)
        reference = self.batch_inputs.masked_image_latents.to(dtype=latents.dtype)
        return edit_mask * latents + (1.0 - edit_mask) * reference

    def _sample_dmd_timesteps(self, batch_size):
        strategy = str(getattr(self.config, "dmd_timestep_strategy", "random")).lower()
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        min_t = max(0, int(getattr(self.config, "dmd_min_timestep", 20)))
        max_t = min(num_train_timesteps - 1, int(getattr(self.config, "dmd_max_timestep", num_train_timesteps - 1)))
        if min_t > max_t:
            raise ValueError(f"Invalid DMD timestep range: min={min_t}, max={max_t}")
        if strategy == "fixed":
            timestep = int(getattr(self.config, "dmd_fixed_timestep", max_t))
            timestep = max(min_t, min(max_t, timestep))
            return torch.full((batch_size,), timestep, dtype=torch.long, device=self.device)
        if strategy == "discrete":
            configured_timesteps = getattr(self.config, "dmd_timesteps", None)
            if configured_timesteps is None:
                raise ValueError("dmd_timestep_strategy='discrete' requires dmd_timesteps to be set.")
            if isinstance(configured_timesteps, str):
                configured_timesteps = [
                    int(item.strip())
                    for item in configured_timesteps.split(",")
                    if item.strip()
                ]
            timesteps = [
                int(t)
                for t in configured_timesteps
                if min_t <= int(t) <= max_t
            ]
            if not timesteps:
                raise ValueError(
                    "No valid dmd_timesteps remain after applying "
                    f"dmd_min_timestep={min_t} and dmd_max_timestep={max_t}."
                )
            timestep_tensor = torch.tensor(timesteps, device=self.device, dtype=torch.long)
            indices = torch.randint(0, timestep_tensor.numel(), (batch_size,), device=self.device)
            return timestep_tensor[indices]
        if strategy != "random":
            raise ValueError(f"Unsupported dmd_timestep_strategy={strategy!r}")
        return torch.randint(min_t, max_t + 1, (batch_size,), device=self.device, dtype=torch.long)

    def _sample_fake_timesteps(self, batch_size):
        num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        min_t = max(0, int(getattr(self.config, "fake_min_timestep", 0)))
        max_t = min(num_train_timesteps - 1, int(getattr(self.config, "fake_max_timestep", num_train_timesteps - 1)))
        if min_t > max_t:
            raise ValueError(f"Invalid fake timestep range: min={min_t}, max={max_t}")
        return torch.randint(min_t, max_t + 1, (batch_size,), device=self.device, dtype=torch.long)

    def forward_generator_latents(self, requires_grad=True):
        z = self._copy_latent_background(self.batch_inputs.z_lq)
        noise_timesteps, model_timesteps = self._select_generator_timesteps(z.shape[0])
        self.last_generator_noise_timestep = noise_timesteps.detach().float().mean()
        self.last_generator_model_timestep = model_timesteps.detach().float().mean()
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=z.device, dtype=torch.float32)
        alpha_prod_t = alphas_cumprod[noise_timesteps]
        model = self.G
        if requires_grad:
            eps = self._unet_pred(model, z, model_timesteps)
        else:
            with torch.no_grad():
                eps = self._unet_pred(model, z, model_timesteps)
        if requires_grad:
            self._capture_generator_attention_map()
        else:
            self.last_generator_attention_map = None
        x0 = self._pred_x0(eps.float(), z.float(), noise_timesteps).to(dtype=z.dtype)
        self.last_generator_stats = {
            "generator_alpha_prod_mean": alpha_prod_t.detach().float().mean(),
            "generator_alpha_prod_min": alpha_prod_t.detach().float().amin(),
            **self._numeric_tensor_stats("G_z_lq", z),
            **self._numeric_tensor_stats("G_prompt_embeds", self.batch_inputs.c_txt["prompt_embeds"]),
            **self._numeric_tensor_stats("G_pooled_prompt_embeds", self.batch_inputs.c_txt["pooled_prompt_embeds"]),
            **self._numeric_tensor_stats("G_add_time_ids", self.batch_inputs.add_time_ids),
            **self._numeric_tensor_stats("G_eps", eps),
            **self._numeric_tensor_stats("G_x_fake", x0),
        }
        if bool(getattr(self.config, "debug_generator_fp32_forward_probe", True)):
            with torch.no_grad():
                eps_fp32 = self._unet_pred_fp32_probe(model, z, model_timesteps)
            self.last_generator_stats.update(self._numeric_tensor_stats("G_eps_fp32_probe", eps_fp32))
        return x0, model_timesteps, eps

    def _capture_generator_attention_map(self):
        self.last_generator_attention_map = None
        if not bool(getattr(self.config, "debug_save_attention_map", False)):
            return
        max_saved = int(getattr(self.config, "debug_max_saved_steps", 200))
        if max_saved >= 0 and int(getattr(self, "debug_saved_steps", 0)) >= max_saved:
            return
        if len(self.cross_attention_scores) == 0:
            return

        mask = self.batch_inputs.mask_latent
        fuse_index = int(getattr(self.config, "fuse_index", 5))
        layer_maps = []
        with torch.no_grad():
            for layer_attention in self.cross_attention_scores.values():
                attention = resize_attn_map_divide2(layer_attention.detach(), mask, fuse_index)
                layer_maps.append(attention.mean(dim=1, keepdim=True))
            if layer_maps:
                self.last_generator_attention_map = (
                    torch.stack(layer_maps, dim=0).mean(dim=0).detach().clamp(0, 1)
                )

    def forward_generator(self):
        x0, _, _ = self.forward_generator_latents(requires_grad=torch.is_grad_enabled())
        x = self._decode_latents(x0)
        return x, x0

    def _dmd_region_mode(self):
        return str(getattr(self.config, "dmd_region_mode", "full")).lower()

    def _dmd_loss_mask(self, like):
        source = str(getattr(self.config, "dmd_mask_source", "effect_mask")).lower()
        if source in ("effect", "effect_mask", "object_effect_mask"):
            mask = getattr(self.batch_inputs, "effect_mask_latent", None)
            if mask is None:
                mask = getattr(self.batch_inputs, "object_effect_mask", None)
        elif source in ("object", "object_mask"):
            mask = getattr(self.batch_inputs, "object_mask", None)
        elif source in ("mask", "mask_latent"):
            mask = getattr(self.batch_inputs, "mask_latent", None)
        else:
            raise ValueError(f"Unsupported dmd_mask_source={source!r}")

        if mask is None:
            mask = getattr(self.batch_inputs, "mask_latent", None)
        if mask is None:
            raise ValueError("DMD masked mode requires an effect/object mask, but none was found.")

        mask = mask.to(device=like.device, dtype=like.dtype)
        if mask.shape[-2:] != like.shape[-2:]:
            mask = F.interpolate(mask.float(), size=like.shape[-2:], mode="nearest").to(dtype=like.dtype)
        if mask.shape[1] != 1:
            mask = mask[:, :1]

        dilate = int(getattr(self.config, "dmd_mask_dilate", 0))
        if dilate > 0:
            kernel_size = 2 * dilate + 1
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=dilate)

        blur = int(getattr(self.config, "dmd_mask_blur", 0))
        if blur > 0:
            kernel_size = 2 * blur + 1
            mask = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=blur)

        return mask.clamp(0, 1)

    def _compute_dmd_loss(self, x_fake):
        bs = x_fake.shape[0]
        noise = torch.randn_like(x_fake)
        timesteps = self._sample_dmd_timesteps(bs)
        self.last_dmd_timestep = timesteps.detach().float().mean()
        dmd_region_mode = self._dmd_region_mode()
        if dmd_region_mode not in ("full", "masked", "effect_mask"):
            raise ValueError(f"Unsupported dmd_region_mode={dmd_region_mode!r}")
        dmd_mask = None if dmd_region_mode == "full" else self._dmd_loss_mask(x_fake)

        with torch.no_grad():
            z_t = self._add_noise_at(x_fake.detach(), noise, timesteps)
            real_pred = self._unet_pred_cfg(
                self.teacher_G,
                z_t,
                timesteps,
                self._cfg_scale("real_CFG", "real_guidance_scale", 1.0),
            )
            fake_pred = self._unet_pred_cfg(
                self.fake_G,
                z_t,
                timesteps,
                self._cfg_scale("fake_CFG", "fake_guidance_scale", 1.0),
            )
            x0_real = self._pred_x0(real_pred.float(), z_t.float(), timesteps)
            x0_fake_score = self._pred_x0(fake_pred.float(), z_t.float(), timesteps)
            p_real = x_fake.detach().float() - x0_real.float()
            p_fake = x_fake.detach().float() - x0_fake_score.float()
            grad = p_real - p_fake
            norm_type = str(getattr(self.config, "dmd_grad_norm_type", "real_abs_mean")).lower()
            if norm_type == "real_abs_mean":
                if dmd_mask is not None and bool(getattr(self.config, "dmd_mask_normalize", True)):
                    mask_f = dmd_mask.float()
                    denom = (p_real.abs() * mask_f).sum(dim=[1, 2, 3], keepdim=True)
                    denom = denom / (mask_f.sum(dim=[1, 2, 3], keepdim=True) * p_real.shape[1] + 1e-8)
                    grad = grad / (denom + 1e-8)
                else:
                    grad = grad / (p_real.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            elif norm_type == "none":
                pass
            else:
                raise ValueError(f"Unsupported dmd_grad_norm_type={norm_type!r}")
            grad_unmasked = torch.nan_to_num(grad)
            if dmd_mask is not None and bool(getattr(self.config, "dmd_masked_grad", True)):
                grad = grad * dmd_mask.float()
            grad = torch.nan_to_num(grad)

        target = (x_fake - grad.to(dtype=x_fake.dtype)).detach()
        if dmd_mask is not None and bool(getattr(self.config, "dmd_loss_normalize_by_mask", True)):
            mask_f = dmd_mask.float()
            per_elem = 0.5 * (x_fake.float() - target.float()).pow(2)
            loss = (per_elem * mask_f).sum() / (mask_f.sum() * x_fake.shape[1] + 1e-8)
        else:
            loss = 0.5 * F.mse_loss(x_fake.float(), target.float(), reduction="mean")
        self.last_dmd_debug = {
            "x_fake": x_fake.detach(),
            "x0_real": x0_real.detach(),
            "x0_fake_score": x0_fake_score.detach(),
            "grad": grad.detach(),
            "grad_unmasked": grad_unmasked.detach(),
            "grad_masked": grad.detach(),
            "z_t": z_t.detach(),
            "timesteps": timesteps.detach(),
        }
        if dmd_mask is not None:
            self.last_dmd_debug["dmd_mask"] = dmd_mask.detach()
        with torch.no_grad():
            grad_abs = grad.detach().abs().mean(dim=1, keepdim=True)
            mask = dmd_mask if dmd_mask is not None else getattr(self.batch_inputs, "effect_mask_latent", None)
            if mask is None:
                mask = getattr(self.batch_inputs, "mask_latent", None)
            if mask is not None:
                mask = mask.to(device=grad_abs.device, dtype=grad_abs.dtype).clamp(0, 1)
                if mask.shape[-2:] != grad_abs.shape[-2:]:
                    mask = F.interpolate(mask, size=grad_abs.shape[-2:], mode="nearest")
                bg_mask = 1.0 - mask
                grad_mask_mean = (grad_abs * mask).sum() / (mask.sum() + 1e-8)
                grad_bg_mean = (grad_abs * bg_mask).sum() / (bg_mask.sum() + 1e-8)
            else:
                grad_mask_mean = torch.tensor(0.0, device=grad_abs.device)
                grad_bg_mean = torch.tensor(0.0, device=grad_abs.device)
            self.last_dmd_stats = {
                "dmd_grad_abs_mean": grad_abs.mean().detach(),
                "dmd_grad_abs_max": grad_abs.amax().detach(),
                "dmd_grad_mask_mean": grad_mask_mean.detach(),
                "dmd_grad_bg_mean": grad_bg_mean.detach(),
                "dmd_grad_mask_bg_ratio": (grad_mask_mean / (grad_bg_mean + 1e-8)).detach(),
                "dmd_p_real_abs_mean": p_real.abs().mean().detach(),
                "dmd_p_fake_abs_mean": p_fake.abs().mean().detach(),
                "dmd_x0_gap_abs_mean": (x0_fake_score.float() - x0_real.float()).abs().mean().detach(),
                "dmd_mask_mean": (
                    dmd_mask.float().mean().detach()
                    if dmd_mask is not None
                    else torch.tensor(0.0, device=grad_abs.device)
                ),
            }
        return loss * getattr(self.config, "lambda_dmd", 1.0), grad

    def _compute_fake_score_loss(self, x_fake_detached):
        bs = x_fake_detached.shape[0]
        noise = torch.randn_like(x_fake_detached)
        timesteps = self._sample_fake_timesteps(bs)
        z_t = self._add_noise_at(x_fake_detached.detach(), noise, timesteps)
        pred = self._unet_pred(self.fake_G, z_t, timesteps)
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x_fake_detached, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        self.last_fake_timestep = timesteps.detach().float().mean()
        return loss * getattr(self.config, "fake_loss_weight", 1.0)

    def _fake_classify_mid_features(self, latents, timesteps):
        model = self.unwrap_model(self.fake_G)
        if model.mid_block is None:
            raise RuntimeError("fake_G has no mid_block for internal GAN classification.")

        captured = {}

        def capture_mid_features(_module, _inputs, output):
            captured["mid_features"] = output[0] if isinstance(output, (tuple, list)) else output

        hook = model.mid_block.register_forward_hook(capture_mid_features)
        unet_output = None
        try:
            unet_output = self._unet_pred(self.fake_G, latents, timesteps)
        finally:
            hook.remove()

        mid_features = captured.get("mid_features", None)
        if mid_features is None:
            raise RuntimeError("Failed to capture fake_G mid_block features for internal GAN classification.")
        mid_features = mid_features.float()
        if isinstance(unet_output, torch.Tensor) and unet_output.requires_grad:
            # Keep the FSDP root output in the autograd graph; the classifier value is unchanged.
            mid_features = mid_features + unet_output.float().mean() * 0.0
        return mid_features

    def _fake_cls_logits(self, latents, timesteps=None):
        if not self._internal_gan_enabled():
            raise RuntimeError("Internal GAN classifier is disabled.")
        if timesteps is None:
            if bool(getattr(self.config, "gan_diffusion", True)):
                max_t = int(getattr(self.config, "gan_diffusion_max_timestep", self.scheduler.config.num_train_timesteps))
                max_t = max(1, min(max_t, int(self.scheduler.config.num_train_timesteps)))
                timesteps = torch.randint(0, max_t, (latents.shape[0],), device=self.device, dtype=torch.long)
                latents = self.scheduler.add_noise(latents, torch.randn_like(latents), timesteps)
            else:
                timesteps = torch.zeros((latents.shape[0],), device=self.device, dtype=torch.long)
        mid_features = self._fake_classify_mid_features(latents, timesteps)
        return self.fake_cls_head(mid_features.float()).flatten()

    def _compute_fake_gan_loss(self, x_fake_detached):
        if not self._internal_gan_enabled() or self.global_step < int(getattr(self.config, "gan_start_step", 0)):
            zero = torch.tensor(0.0, device=self.device, dtype=x_fake_detached.dtype)
            return zero, zero, zero
        logits = self._fake_cls_logits(torch.cat([self.batch_inputs.z_gt.detach(), x_fake_detached.detach()], dim=0))
        real_logits, fake_logits = logits.chunk(2, dim=0)
        loss_real = F.softplus(-real_logits).mean()
        loss_fake = F.softplus(fake_logits).mean()
        loss = (loss_real + loss_fake) * getattr(self.config, "guidance_cls_loss_weight", 1.0)
        return loss, real_logits.detach().mean(), fake_logits.detach().mean()

    def _compute_generator_internal_gan_loss(self, x_fake):
        gen_gan_start_step = int(
            getattr(self.config, "gen_gan_start_step", getattr(self.config, "gan_start_step", 0))
        )
        if not self._internal_gan_enabled() or self.global_step < gen_gan_start_step:
            return torch.tensor(0.0, device=self.device, dtype=x_fake.dtype)
        fake_g_training = self.fake_G.training
        fake_cls_training = self.fake_cls_head.training
        self.fake_G.eval()
        self.fake_cls_head.eval()
        try:
            logits = self._fake_cls_logits(x_fake)
            loss = F.softplus(-logits).mean() * getattr(
                self.config, "gen_cls_loss_weight", getattr(self.config, "lambda_gan", 0.0)
            )
        finally:
            self.fake_G.train(fake_g_training)
            self.fake_cls_head.train(fake_cls_training)
        return loss

    def _compute_train_objectclear_style_loss(self):
        z_gt = self.batch_inputs.z_gt.to(dtype=self.weight_dtype)
        mask = self.batch_inputs.mask_latent.to(dtype=self.weight_dtype)
        bs = z_gt.shape[0]

        noise = torch.randn_like(z_gt)
        if str(getattr(self.config, "scheduler_timestep_spacing", "trailing")).lower() == "fixed":
            max_noise_timestep = int(self.scheduler.config.num_train_timesteps) - 1
            noise_timesteps = torch.randint(
                0,
                max_noise_timestep + 1,
                (bs,),
                device=self.device,
                dtype=torch.long,
            )
            model_max_timestep = getattr(self.config, "generator_timestep", None)
            if model_max_timestep is None:
                model_max_timestep = self._get_configured_noise_timestep()
            model_max_timestep = int(model_max_timestep)
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)
            alpha = alphas_cumprod.clamp(1e-8, 1.0 - 1e-8)
            log_snr = torch.log(alpha) - torch.log1p(-alpha)
            log_snr_start = log_snr[0]
            log_snr_end = log_snr[max_noise_timestep]
            noise_progress = (log_snr_start - log_snr[noise_timesteps]) / (
                log_snr_start - log_snr_end + 1e-8
            )
            model_timesteps = (noise_progress.clamp(0.0, 1.0) * float(model_max_timestep)).round().long()
            model_timesteps = model_timesteps.clamp(0, model_max_timestep)
        else:
            noise_timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (bs,),
                device=self.device,
                dtype=torch.long,
            )
            model_timesteps = noise_timesteps

        noisy_latents = self.scheduler.add_noise(z_gt, noise, noise_timesteps)
        noisy_model_input = self.scheduler.scale_model_input(noisy_latents, model_timesteps)
        noisy_model_input = torch.cat([noisy_model_input, mask, self.batch_inputs.masked_image_latents], dim=1)

        with self._unet_forward_context():
            model_pred = self.G(
                noisy_model_input,
                model_timesteps,
                encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                added_cond_kwargs={
                    "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                    "time_ids": self.batch_inputs.add_time_ids,
                },
            ).sample

        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(z_gt, noise, noise_timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.last_diffusion_noise_timestep = noise_timesteps.detach().float().mean()
        self.last_diffusion_model_timestep = model_timesteps.detach().float().mean()

        localization_loss = torch.tensor(0.0, device=self.device, dtype=diffusion_loss.dtype)
        if getattr(self.config, "object_localization", False) and self.object_localization_loss_fn is not None:
            fuse_index = getattr(self.config, "fuse_index", 5)
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                self.batch_inputs.object_effect_mask,
                self.object_localization_loss_fn,
                fuse_index,
            )
            clear_cross_attention_scores(self.cross_attention_scores)

        return diffusion_loss, localization_loss

    def _generator_external_d_loss(self, x_pred, z_pred):
        if not self._external_d_enabled() or not hasattr(self, "D"):
            return torch.tensor(0.0, device=self.device, dtype=x_pred.dtype)
        if self.global_step < int(getattr(self.config, "gan_start_step", 0)):
            return torch.tensor(0.0, device=self.device, dtype=x_pred.dtype)
        self.unwrap_model(self.D).eval().requires_grad_(False)
        if getattr(self.config, "use_D_sdxl", False):
            return self.D(
                latents=z_pred,
                mask=self.batch_inputs.mask_latent,
                reference_latents=self.batch_inputs.masked_image_latents,
                encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                added_cond_kwargs={
                    "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                    "time_ids": self.batch_inputs.add_time_ids,
                },
                for_G=True,
            ).mean() * getattr(self.config, "lambda_gan", 0.0)
        return self.D(x_pred, for_G=True).mean() * getattr(self.config, "lambda_gan", 0.0)

    def optimize_generator(self):
        fake_probe_before = self._fake_weight_probe()
        with self.accelerator.accumulate(self.G):
            if self._external_d_enabled() and hasattr(self, "D"):
                self.unwrap_model(self.D).eval().requires_grad_(False)
            x_fake, _, _ = self.forward_generator_latents(requires_grad=True)
            dmd_enabled = self._dmd_enabled()
            if not dmd_enabled:
                loss_dmd = torch.tensor(0.0, device=self.device, dtype=x_fake.dtype)
                grad = torch.zeros_like(x_fake)
            else:
                loss_dmd, grad = self._compute_dmd_loss(x_fake)
            x_pred = self._decode_latents(x_fake)
            self.G_pred = x_pred.detach()
            self.G_pred_latents = x_fake.detach()

            effect_mask = self.batch_inputs.object_effect_mask
            bg_mask = 1.0 - effect_mask
            zero = self._zero_like_loss(loss_dmd)
            lambda_gt_latent = float(getattr(self.config, "lambda_gt_latent", 0.0))
            lambda_l2_latent = float(getattr(self.config, "lambda_l2_latent", 0.0))
            lambda_l2 = float(getattr(self.config, "lambda_l2", 0.0))
            lambda_mask_l1 = float(getattr(self.config, "lambda_mask_l1", 0.0))
            lambda_mask_background = float(getattr(self.config, "lambda_mask_background", 0.0))
            lambda_lpips_gt = float(getattr(self.config, "lambda_lpips_gt", 0.0))

            loss_gt_latent = (
                F.mse_loss(x_fake.float(), self.batch_inputs.z_gt.float()) * lambda_gt_latent
                if lambda_gt_latent > 0.0
                else zero
            )
            loss_l2_latent = (
                F.mse_loss(x_fake.float(), self.batch_inputs.z_gt.float(), reduction="mean") * lambda_l2_latent
                if lambda_l2_latent > 0.0
                else zero
            )
            loss_l2 = (
                F.mse_loss(x_pred.float(), self.batch_inputs.gt.float(), reduction="mean") * lambda_l2
                if lambda_l2 > 0.0
                else zero
            )
            loss_mask_l1 = (
                self._masked_l1(x_pred, self.batch_inputs.gt, effect_mask) * lambda_mask_l1
                if lambda_mask_l1 > 0.0
                else zero
            )
            loss_mask_background = (
                self._masked_l1(x_pred, self.batch_inputs.gt, bg_mask) * lambda_mask_background
                if lambda_mask_background > 0.0
                else zero
            )
            lpips_per_sample = None
            if lambda_lpips_gt > 0.0:
                if bool(getattr(self.config, "local_lpips", False)):
                    raw_lpips = self._lpips_gt_loss(x_pred, self.batch_inputs.gt)
                    lpips_per_sample = raw_lpips.detach().reshape(1)
                else:
                    lpips_per_sample = self.net_lpips(x_pred.float(), self.batch_inputs.gt.float()).view(-1)
                    raw_lpips = lpips_per_sample.mean()
                loss_lpips = raw_lpips * lambda_lpips_gt
            else:
                loss_lpips = zero

            lambda_teacher_image_l1 = 0.0 if not dmd_enabled else float(getattr(self.config, "lambda_teacher_image_l1", 0.0))
            if lambda_teacher_image_l1 > 0.0:
                teacher_img = self._decode_latents(self.last_dmd_debug["x0_real"]).detach()
                loss_teacher_l1 = F.l1_loss(x_pred.float(), teacher_img.float(), reduction="mean") * lambda_teacher_image_l1
            else:
                loss_teacher_l1 = zero

            lambda_diffusion = float(getattr(self.config, "lambda_diffusion", 0.0))
            lambda_localization = float(getattr(self.config, "object_localization_weight", 0.0))
            if lambda_diffusion > 0.0 or (
                getattr(self.config, "object_localization", False) and lambda_localization > 0.0
            ):
                diffusion_loss, localization_loss = self._compute_train_objectclear_style_loss()
            else:
                diffusion_loss, localization_loss = zero, zero
            loss_diffusion = diffusion_loss * lambda_diffusion
            loss_localization = localization_loss * lambda_localization
            if not dmd_enabled:
                loss_gan = zero
                loss_external_D = zero
            else:
                loss_gan = self._compute_generator_internal_gan_loss(x_fake)
                loss_external_D = self._generator_external_d_loss(x_pred, x_fake)

            loss = (
                loss_dmd
                + loss_gt_latent
                + loss_l2_latent
                + loss_l2
                + loss_mask_l1
                + loss_mask_background
                + loss_teacher_l1
                + loss_lpips
                + loss_diffusion
                + loss_localization
                + loss_gan
                + loss_external_D
            )

            lpips_trace_enabled = self._lpips_trace_enabled()
            trace_params = self._optimizer_params(getattr(self, "G_opt", None))
            params_before_step = self._debug_param_snapshot(trace_params) if trace_params else []
            probe_before_step = self._param_update_probe(trace_params) if trace_params else []
            self.accelerator.backward(loss)
            grad_before_clip = self._debug_grad_stats(trace_params) if lpips_trace_enabled else {}
            if self.accelerator.sync_gradients:
                clip_return = self._clip_grad_norm(self.G, trace_params, self.config.max_grad_norm)
            else:
                clip_return = torch.tensor(0.0, device=self.device)
            grad_after_clip = self._debug_grad_stats(trace_params) if lpips_trace_enabled else {}
            self.G_opt.step()
            params_after_step = self._debug_param_snapshot(trace_params) if trace_params else []
            probe_after_step = self._param_update_probe(trace_params) if trace_params else []
            param_delta_log = self._debug_delta_summary(
                self._debug_param_delta(params_before_step, params_after_step),
                "G_opt",
            )
            param_delta_log.update(
                self._param_update_probe_delta_summary(probe_before_step, probe_after_step, "G_opt")
            )
            self.G_opt.zero_grad()
            if hasattr(self, "fake_opt"):
                self.fake_opt.zero_grad()
            if lpips_trace_enabled:
                trace_step = int(self.global_step) + 1
                self._write_lpips_trace(
                    trace_step,
                    {
                        "loss": loss,
                        "loss_lpips": loss_lpips,
                        "loss_localization": loss_localization,
                        "loss_diffusion": loss_diffusion,
                        "loss_dmd": loss_dmd,
                        "loss_l2": loss_l2,
                        "loss_mask_l1": loss_mask_l1,
                        "loss_mask_background": loss_mask_background,
                    },
                    lpips_per_sample,
                    params_before_step,
                    params_after_step,
                    grad_before_clip,
                    grad_after_clip,
                    clip_return,
                )

        if hasattr(self, "fake_G"):
            self.unwrap_model(self.fake_G).train()
        fake_probe_after = self._fake_weight_probe()
        fake_probe_log = {}
        if fake_probe_before and fake_probe_after:
            before = fake_probe_before["fake_probe_checksum"]
            after = fake_probe_after["fake_probe_checksum"]
            fake_probe_log = {
                "fake_probe_checksum_after_G": after,
                "fake_probe_abs_mean_after_G": fake_probe_after["fake_probe_abs_mean"],
                "fake_probe_delta_during_G": (after - before).abs(),
            }
        generator_stats = {
            "generator_noise_timestep": getattr(
                self, "last_generator_noise_timestep", torch.tensor(0.0, device=self.device)
            ),
            "generator_model_timestep": getattr(
                self, "last_generator_model_timestep", torch.tensor(0.0, device=self.device)
            ),
            **getattr(self, "last_generator_stats", {}),
            **self._numeric_tensor_stats("G_gt", self.batch_inputs.gt),
            **self._numeric_tensor_stats("G_z_gt", self.batch_inputs.z_gt),
            **self._numeric_tensor_stats("G_masked_image_latents", self.batch_inputs.masked_image_latents),
            **self._numeric_tensor_stats("G_x_pred", self.G_pred),
            **self._numeric_tensor_stats("G_loss_lpips", loss_lpips),
            **self._numeric_tensor_stats("G_loss_total", loss),
            **self._model_param_stats("G", self.G),
            **param_delta_log,
        }
        return {
            "G_total": loss,
            "G_dmd": loss_dmd,
            "G_gt_latent": loss_gt_latent,
            "G_l2_latent": loss_l2_latent,
            "G_l2": loss_l2,
            "G_mask_l1": loss_mask_l1,
            "G_mask_background": loss_mask_background,
            "G_teacher_l1": loss_teacher_l1,
            "G_lpips": loss_lpips,
            "G_diffusion": loss_diffusion,
            "G_localization": loss_localization,
            "G_gan": loss_gan,
            "G_external_D": loss_external_D,
            "dmd_timestep": getattr(self, "last_dmd_timestep", torch.tensor(0.0, device=self.device)),
            "diffusion_noise_timestep": getattr(
                self, "last_diffusion_noise_timestep", torch.tensor(0.0, device=self.device)
            ),
            "diffusion_model_timestep": getattr(
                self, "last_diffusion_model_timestep", torch.tensor(0.0, device=self.device)
            ),
            "dmd_grad_norm": grad.float().norm(),
            **getattr(self, "last_dmd_stats", {}),
            **fake_probe_log,
            **generator_stats,
        }

    def optimize_fake_model(self):
        with torch.no_grad():
            x_fake, _, _ = self.forward_generator_latents(requires_grad=False)
        fake_probe_before = self._fake_weight_probe()
        probe_param = self._fake_trainable_param_slice()
        if probe_param is not None:
            probe_slice_before = probe_param.detach().flatten()[:1024].float().cpu().clone()
        else:
            probe_slice_before = None
        fake_grad_log = {}
        with self.accelerator.accumulate(self.fake_G):
            self.unwrap_model(self.fake_G).train()
            if self._internal_gan_enabled():
                self.fake_cls_head.train().requires_grad_(True)
            loss_fake_score = self._compute_fake_score_loss(x_fake.detach())
            loss_fake_gan, real_logits, fake_logits = self._compute_fake_gan_loss(x_fake.detach())
            loss_fake = loss_fake_score + loss_fake_gan
            self.accelerator.backward(loss_fake)
            fake_grad_log = self._fake_grad_probe()
            if self.accelerator.sync_gradients:
                fake_clip_models = [self.fake_G]
                if self._internal_gan_enabled() and hasattr(self, "fake_cls_head"):
                    fake_clip_models.append(self.fake_cls_head)
                self._clip_grad_norm(fake_clip_models, self.fake_params, self.config.max_grad_norm)
            self.fake_opt.step()
            self.fake_opt.zero_grad()
            self.G_opt.zero_grad()
        fake_probe_after = self._fake_weight_probe()
        fake_probe_log = {}
        if fake_probe_before and fake_probe_after:
            before = fake_probe_before["fake_probe_checksum"]
            after = fake_probe_after["fake_probe_checksum"]
            fake_probe_log = {
                "fake_probe_checksum_after_fake": after,
                "fake_probe_abs_mean_after_fake": fake_probe_after["fake_probe_abs_mean"],
                "fake_probe_delta_during_fake": (after - before).abs(),
            }
        if probe_param is not None and probe_slice_before is not None:
            probe_slice_after = probe_param.detach().flatten()[:1024].float().cpu()
            probe_slice_delta = (probe_slice_after - probe_slice_before).abs()
            fake_probe_log.update(
                {
                    "fake_probe_first_param_delta_max": torch.tensor(
                        float(probe_slice_delta.max().item()), device=self.device
                    ),
                    "fake_probe_first_param_delta_mean": torch.tensor(
                        float(probe_slice_delta.mean().item()), device=self.device
                    ),
                }
            )
        return {
            "fake_total": loss_fake,
            "fake_score": loss_fake_score,
            "fake_gan": loss_fake_gan,
            "fake_gan_logits_real": real_logits,
            "fake_gan_logits_fake": fake_logits,
            "fake_timestep": getattr(self, "last_fake_timestep", torch.tensor(0.0, device=self.device)),
            **fake_grad_log,
            **fake_probe_log,
        }

    def optimize_discriminator(self):
        if not self._external_d_enabled() or not hasattr(self, "D"):
            return {}
        if self.global_step < int(getattr(self.config, "gan_start_step", 0)):
            return {}
        gt = self.batch_inputs.gt
        z_gt = self.batch_inputs.z_gt
        with torch.no_grad():
            x_fake, _, _ = self.forward_generator_latents(requires_grad=False)
            x = self._decode_latents(x_fake)
        self.G_pred = x.detach()
        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            if getattr(self.config, "use_D_sdxl", False):
                loss_D_real, real_logits = self.D(
                    z_gt,
                    mask=self.batch_inputs.mask_latent,
                    reference_latents=self.batch_inputs.masked_image_latents,
                    encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                    added_cond_kwargs={
                        "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                        "time_ids": self.batch_inputs.add_time_ids,
                    },
                    for_real=True,
                    return_logits=True,
                )
                loss_D_fake, fake_logits = self.D(
                    x_fake.detach(),
                    mask=self.batch_inputs.mask_latent,
                    reference_latents=self.batch_inputs.masked_image_latents,
                    encoder_hidden_states=self.batch_inputs.c_txt["prompt_embeds"],
                    added_cond_kwargs={
                        "text_embeds": self.batch_inputs.c_txt["pooled_prompt_embeds"],
                        "time_ids": self.batch_inputs.add_time_ids,
                    },
                    for_real=False,
                    return_logits=True,
                )
            else:
                loss_D_real, real_logits = self.D(gt, for_real=True, return_logits=True)
                loss_D_fake, fake_logits = self.D(x.detach(), for_real=False, return_logits=True)
            loss_D = (loss_D_real.mean() + loss_D_fake.mean()) * getattr(self.config, "lambda_D", 1.0)
            self.accelerator.backward(loss_D)
            if self.accelerator.sync_gradients:
                self._clip_grad_norm(self.D, self.D_params, self.config.max_grad_norm)
            self.D_opt.step()
            self.D_opt.zero_grad()
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        return {"D": loss_D, "D_logits_real": real_logits, "D_logits_fake": fake_logits}

    def _debug_save_step_images(self):
        if not bool(getattr(self.config, "debug_save_every_step", False)):
            return
        max_saved = int(getattr(self.config, "debug_max_saved_steps", 200))
        if max_saved >= 0 and self.debug_saved_steps >= max_saved:
            return
        save_all_processes = bool(getattr(self.config, "debug_save_all_processes", False))
        if not save_all_processes and not self.accelerator.is_main_process:
            return

        debug_dir = getattr(self.config, "debug_image_dir", None)
        if debug_dir is None:
            debug_dir = os.path.join(self.config.output_dir, "debug_generator_steps")
        step_dir = os.path.join(debug_dir, f"{self.global_step:07}")
        process_index = int(getattr(self.accelerator, "process_index", 0))
        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        if save_all_processes and num_processes > 1:
            step_dir = os.path.join(step_dir, f"rank_{process_index:03d}")
        os.makedirs(step_dir, exist_ok=True)

        with torch.no_grad():
            debug = getattr(self, "last_dmd_debug", {})
            if not hasattr(self, "batch_inputs") or not hasattr(self, "G_pred"):
                return

            batch_size = int(self.G_pred.shape[0])
            if not bool(getattr(self.config, "debug_save_all_samples", False)):
                batch_size = min(batch_size, 1)
            output_size = self.G_pred.shape[-2:]
            prompts = getattr(self.batch_inputs, "prompt", [])
            timesteps = debug.get("timesteps")

            for sample_index in range(batch_size):
                sample_dir = os.path.join(step_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)

                image_tensors = {
                    "input_img": self.batch_inputs.input_img[sample_index : sample_index + 1],
                    "gt_img": self.batch_inputs.gt[sample_index : sample_index + 1],
                    "input_mask": self.batch_inputs.object_mask[sample_index : sample_index + 1],
                    "unet_output_img": self.G_pred[sample_index : sample_index + 1],
                    "noisy_img": self._decode_latents(
                        self.batch_inputs.z_lq[sample_index : sample_index + 1]
                    ),
                    "student_noisy_img": (
                        self._decode_latents(debug["z_t"][sample_index : sample_index + 1])
                        if "z_t" in debug
                        else None
                    ),
                    "fake_unet_output_img": (
                        self._decode_latents(debug["x0_fake_score"][sample_index : sample_index + 1])
                        if "x0_fake_score" in debug
                        else None
                    ),
                    "real_unet_output_img": (
                        self._decode_latents(debug["x0_real"][sample_index : sample_index + 1])
                        if "x0_real" in debug
                        else None
                    ),
                }
                for name, image in image_tensors.items():
                    if image is None:
                        continue
                    self._debug_save_image_tensor(
                        image,
                        os.path.join(sample_dir, f"{name}.png"),
                        mask=name == "input_mask",
                    )

                dmd_mask = debug.get("dmd_mask")
                if dmd_mask is not None:
                    mask_vis = F.interpolate(
                        dmd_mask[sample_index : sample_index + 1].float(),
                        size=output_size,
                        mode="nearest",
                    )[0, 0].clamp(0, 1)
                    self._debug_save_grayscale_tensor(mask_vis, os.path.join(sample_dir, "dmd_mask.png"))

                effect_mask_vis = F.interpolate(
                    self.batch_inputs.object_effect_mask[sample_index : sample_index + 1].float(),
                    size=output_size,
                    mode="nearest",
                )[0, 0].clamp(0, 1)
                self._debug_save_grayscale_tensor(
                    effect_mask_vis,
                    os.path.join(sample_dir, "object_effect_mask.png"),
                )

                attention_maps = getattr(self, "last_generator_attention_map", None)
                if isinstance(attention_maps, torch.Tensor) and sample_index < attention_maps.shape[0]:
                    attention_vis = F.interpolate(
                        attention_maps[sample_index : sample_index + 1].float(),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )[0, 0].clamp(0, 1)
                    attention_color = self._debug_turbo_colormap(attention_vis)
                    self._debug_save_image_tensor(
                        attention_color * 2.0 - 1.0,
                        os.path.join(sample_dir, "attention_map_color.png"),
                    )

                grad_unmasked = debug.get("grad_unmasked", debug.get("grad"))
                grad_masked = debug.get("grad_masked", debug.get("grad"))
                gradient_scale = 0.0
                if grad_unmasked is not None and grad_masked is not None:
                    raw_magnitude = grad_unmasked[sample_index].float().abs().mean(dim=0, keepdim=True)
                    masked_magnitude = grad_masked[sample_index].float().abs().mean(dim=0, keepdim=True)
                    gradient_scale = float(raw_magnitude.amax().cpu().item())
                    scale = raw_magnitude.amax().clamp_min(1e-8)
                    raw_vis = F.interpolate(
                        (raw_magnitude / scale).unsqueeze(0),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )[0, 0].clamp(0, 1)
                    masked_vis = F.interpolate(
                        (masked_magnitude / scale).unsqueeze(0),
                        size=output_size,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )[0, 0].clamp(0, 1)
                    self._debug_save_grayscale_tensor(
                        raw_vis,
                        os.path.join(sample_dir, "dmd_gradient_gray.png"),
                    )
                    self._debug_save_grayscale_tensor(
                        masked_vis,
                        os.path.join(sample_dir, "masked_dmd_gradient_gray.png"),
                    )

                timestep = None
                if isinstance(timesteps, torch.Tensor) and sample_index < timesteps.numel():
                    timestep = int(timesteps.flatten()[sample_index].cpu().item())
                metadata = {
                    "global_step": int(self.global_step),
                    "rank": process_index,
                    "sample_index_on_rank": sample_index,
                    "prompt": prompts[sample_index] if sample_index < len(prompts) else "",
                    "dmd_timestep": timestep,
                    "noisy_img_source": "Initial Gaussian latent z_lq provided to the student UNet.",
                    "student_noisy_img_source": "Student output diffused to the sampled DMD timestep.",
                    "attention_map": {
                        "source": "Student generator forward cross-attention for the configured object token.",
                        "token_index": int(getattr(self.config, "fuse_index", 5)),
                        "aggregation": "Per-layer head mean followed by mean over captured layers.",
                        "colormap": "turbo",
                    },
                    "gradient_visualization": {
                        "value": "mean absolute DMD gradient over latent channels",
                        "shared_scale_max": gradient_scale,
                        "normalization": "Both gradient PNGs use the unmasked gradient maximum for this sample.",
                    },
                }
                with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
        self.debug_saved_steps += 1

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        validation_steps = int(getattr(self.config, "validation_steps", 0))

        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                loss_dict = {}

                if not self._dmd_enabled():
                    compute_generator = True
                else:
                    fake_loss = self.optimize_fake_model()
                    loss_dict.update(fake_loss)
                    update_ratio = max(1, int(getattr(self.config, "fake_update_ratio", 5)))
                    compute_generator = (self.batch_count % update_ratio) == 0
                if compute_generator:
                    loss_dict.update(self.optimize_generator())
                    if self._external_d_enabled():
                        loss_dict.update(self.optimize_discriminator())

                for key, value in loss_dict.items():
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(float(value), device=self.device)
                    value = value.detach()
                    if value.layout != torch.strided:
                        value = value.to_dense()
                    value = value.float().mean().reshape(1).to(self.device).contiguous()
                    avg_loss = self.accelerator.gather(value).mean()
                    train_loss[key] = avg_loss.item()

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    if compute_generator:
                        self.ema_handler.update()
                        self.global_step += 1
                        self.pbar.update(1)
                        _, _, peak = print_vram_state(None)
                        self.pbar.set_description(f"DMD Generator Step, VRAM peak: {peak:.2f} GB")

                        log_dict = {f"loss/{key}": value for key, value in train_loss.items()}
                        self.accelerator.log(log_dict, step=self.global_step)
                        self.log_metrics_csv(log_dict, self.global_step)
                        self._debug_save_step_images()

                        if self.config.use_vae and (self.global_step % self.config.log_image_steps == 0 or self.global_step == 1):
                            self.log_images()
                        disable_debug_ckpt = bool(getattr(self.config, "debug_lpips_trace_disable_checkpoint", False))
                        if not disable_debug_ckpt and (
                            self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1
                        ):
                            self.save_checkpoint()
                        disable_debug_val = bool(getattr(self.config, "debug_lpips_trace_disable_validation", False))
                        if not disable_debug_val and validation_steps > 0 and (
                            self.global_step % validation_steps == 0 or self.global_step == 1
                        ):
                            self.validate()
                    train_loss = {}

                if self.global_step >= self.config.max_train_steps:
                    break
        self.accelerator.end_training()

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.output_dir)
                checkpoints = [
                    d
                    for d in checkpoints
                    if d.startswith("checkpoint-") and d != "checkpoint-0" and d.split("-")[-1].isdigit()
                ]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        shutil.rmtree(os.path.join(self.config.output_dir, removing_checkpoint))
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        if self._is_fsdp_training():
            self._save_fsdp_model_only_checkpoint(save_path)
        else:
            self.accelerator.save_state(save_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            logger.info(f"Saved state to {save_path}")
            self.ema_handler.save_ema_weights(save_path)
            logger.info(f"Saved ema weights to {save_path}")

    @torch.no_grad()
    def validate(self):
        if self.val_dataloader is None:
            return
        self.accelerator.wait_for_everyone()

        self._clear_transient_cuda_tensors()
        self.G.eval()
        validation_hook_installed = self._enable_validation_attention_hook_if_needed()
        if self.config.use_ema:
            self.ema_handler.activate_ema_weights()

        metric_names = [
            "unfused_masked_psnr",
            "unfused_full_psnr",
            "unfused_lpips",
            "fused_masked_psnr",
            "fused_full_psnr",
            "fused_lpips",
        ]
        metrics = {name: 0.0 for name in metric_names}
        metric_counts = {name: 0 for name in metric_names}
        num_batches = 0
        num_samples = 0
        save_batches = int(getattr(self.config, "validation_save_batches", 4))
        val_image_root = getattr(
            self.config,
            "validation_image_dir",
            os.path.join(self.config.output_dir, "validation", "images"),
        )
        val_save_dir = os.path.join(val_image_root, f"{self.global_step:07}")
        if self.accelerator.is_main_process:
            os.makedirs(val_save_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        for batch_idx, batch in enumerate(self.val_dataloader):
            max_batches = int(getattr(self.config, "validation_max_batches", -1))
            if max_batches > 0 and batch_idx >= max_batches:
                break
            self.prepare_batch_inputs(batch)
            if bool(getattr(self.config, "apply_attention_guided_fusion_in_validation", False)):
                clear_cross_attention_scores(self.cross_attention_scores)
            unfused_x, unfused_z = self.forward_generator()
            fused_x, _ = self._validation_attention_guided_fusion(unfused_x, unfused_z)
            if bool(getattr(self.config, "apply_attention_guided_fusion_in_validation", False)):
                clear_cross_attention_scores(self.cross_attention_scores)
            unfused_pred = ((unfused_x + 1.0) / 2.0).clamp(0, 1)
            fused_pred = ((fused_x + 1.0) / 2.0).clamp(0, 1)
            gt = ((self.batch_inputs.gt + 1.0) / 2.0).clamp(0, 1)
            input_img = ((self.batch_inputs.input_img + 1.0) / 2.0).clamp(0, 1)
            mask = self.batch_inputs.object_mask.clamp(0, 1)
            effect_mask = self.batch_inputs.object_effect_mask.clamp(0, 1)

            bs = unfused_pred.shape[0]
            for pred_name, pred_x in (("unfused", unfused_x), ("fused", fused_x)):
                masked_psnr_sum = 0.0
                full_psnr_sum = 0.0
                for sample_idx in range(bs):
                    masked_psnr_sum += self._masked_psnr(
                        pred_x[sample_idx : sample_idx + 1],
                        self.batch_inputs.gt[sample_idx : sample_idx + 1],
                        effect_mask[sample_idx : sample_idx + 1],
                    )
                    full_psnr_sum += self._full_psnr(
                        pred_x[sample_idx : sample_idx + 1],
                        self.batch_inputs.gt[sample_idx : sample_idx + 1],
                    )
                metrics[f"{pred_name}_masked_psnr"] += masked_psnr_sum
                metrics[f"{pred_name}_full_psnr"] += full_psnr_sum
                metric_counts[f"{pred_name}_masked_psnr"] += bs
                metric_counts[f"{pred_name}_full_psnr"] += bs

                lpips_val = self.net_lpips(pred_x.float(), self.batch_inputs.gt.float()).mean().item()
                metrics[f"{pred_name}_lpips"] += lpips_val * bs
                metric_counts[f"{pred_name}_lpips"] += bs
            num_batches += 1
            num_samples += bs

            if self.accelerator.is_main_process and batch_idx < save_batches:
                for sample_idx in range(bs):
                    for pred_name, pred in (("unfused", unfused_pred), ("fused", fused_pred)):
                        pred_dir = os.path.join(val_save_dir, pred_name)
                        os.makedirs(pred_dir, exist_ok=True)
                        image_arr = (
                            pred[sample_idx].permute(1, 2, 0).cpu().numpy() * 255.0
                        ).clip(0, 255).astype(np.uint8)
                        Image.fromarray(image_arr).save(
                            os.path.join(pred_dir, f"batch{batch_idx}_sample{sample_idx}_{pred_name}.png")
                        )

                    grid = make_grid(
                        [
                            input_img[sample_idx].cpu(),
                            gt[sample_idx].cpu(),
                            unfused_pred[sample_idx].cpu(),
                            fused_pred[sample_idx].cpu(),
                            mask[sample_idx].repeat(3, 1, 1).cpu(),
                        ],
                        nrow=5,
                    )
                    image_arr = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    Image.fromarray(image_arr).save(
                        os.path.join(val_save_dir, f"batch{batch_idx}_sample{sample_idx}_input_gt_unfused_fused_mask.png")
                    )

        log_metrics = {}
        if num_samples > 0:
            log_metrics = {
                f"validation/{key}": metrics[key] / max(metric_counts[key], 1)
                for key in metric_names
            }
            if self.accelerator.is_main_process:
                self.accelerator.log(log_metrics, step=self.global_step)
                logger.info(f"Validation at step {self.global_step}: {log_metrics}")
        if self.accelerator.is_main_process:
            self._append_validation_csv(log_metrics, num_batches, num_samples, val_save_dir)

        if self.config.use_ema:
            self.ema_handler.deactivate_ema_weights()
        if validation_hook_installed:
            self._restore_attention_score_hooks(self.G)
        self.G.train()
        self._clear_transient_cuda_tensors()
        self.accelerator.wait_for_everyone()

    def _append_validation_csv(self, log_metrics, num_batches, num_samples, val_save_dir):
        if not self.accelerator.is_main_process:
            return
        csv_path = getattr(
            self.config,
            "validation_metrics_csv",
            os.path.join(self.config.output_dir, "validation", "metrics.csv"),
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        row = {
            "step": int(self.global_step),
            "num_batches": int(num_batches),
            "num_samples": int(num_samples),
            "image_dir": val_save_dir,
        }
        row.update({key.replace("validation/", ""): float(value) for key, value in log_metrics.items()})
        base_fields = ["step", "num_batches", "num_samples"]
        tail_fields = ["image_dir"]
        metric_fields = [key for key in row.keys() if key not in base_fields and key not in tail_fields]
        desired_fields = base_fields + metric_fields + tail_fields

        existing_rows = []
        existing_fields = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = [
                    {key: value for key, value in row.items() if key is not None}
                    for row in reader
                ]

        extra_existing_fields = [
            field
            for field in existing_fields
            if field not in desired_fields and field not in tail_fields
        ]
        fieldnames = base_fields + metric_fields + extra_existing_fields + tail_fields
        should_rewrite = bool(existing_rows or existing_fields) and existing_fields != fieldnames
        mode = "w" if should_rewrite or not os.path.exists(csv_path) else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow(existing_row)
            writer.writerow(row)

    def log_images(self):
        if not hasattr(self, "G_pred"):
            return
        n = min(4, self.G_pred.shape[0])
        image_logs = {
            "input": (self.batch_inputs.input_img[:n] + 1) / 2,
            "gt": (self.batch_inputs.gt[:n] + 1) / 2,
            "student_dmd_1step": (self.G_pred[:n] + 1) / 2,
        }
        debug = getattr(self, "last_dmd_debug", {})
        if "x0_real" in debug:
            with torch.no_grad():
                image_logs["dmd_real_x0"] = (self._decode_latents(debug["x0_real"][:n]) + 1) / 2
        if "x0_fake_score" in debug:
            with torch.no_grad():
                image_logs["dmd_fake_x0"] = (self._decode_latents(debug["x0_fake_score"][:n]) + 1) / 2

        if not self.accelerator.is_main_process:
            return
        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(self.config.output_dir, self.config.logging_dir, "log_images", f"{self.global_step:07}", key)
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))
