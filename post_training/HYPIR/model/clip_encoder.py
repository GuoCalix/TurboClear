from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers.models.clip.modeling_clip import (
    CLIPPreTrainedModel,
    CLIPModel,
)


@contextmanager
def _allow_scalar_torch_empty():
    original_empty = torch.empty

    def empty_compat(*args, **kwargs):
        if len(args) == 0:
            return original_empty((), **kwargs)
        return original_empty(*args, **kwargs)

    torch.empty = empty_compat
    try:
        yield
    finally:
        torch.empty = original_empty


class CLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        cache_dir
    ):
        with _allow_scalar_torch_empty():
            model = CLIPModel.from_pretrained(
                global_model_name_or_path,
                subfolder="image_prompt_encoder",
                cache_dir=cache_dir,
                low_cpu_mem_usage=False,
            )
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return CLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        b, c, h, w = object_pixel_values.shape

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, 1, -1)
        return object_embeds
