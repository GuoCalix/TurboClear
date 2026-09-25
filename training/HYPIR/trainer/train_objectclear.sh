accelerate launch --main_process_port=18836 train_objectclear.py \
--pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
--image_encoder_name_or_path openai/clip-vit-large-patch14 \
--output_dir="./exp/3.27_train" \
--validation_file ./davis_validation/validation.jsonl \
--image_dir1 ./data/cropped_image/captured \
--image_dir2 ./data/cropped_image/synthetic/shadow_single_object \
--image_dir3 ./data/cropped_image/synthetic/shadow_multi_object \
--image_dir4 ./data/cropped_image/synthetic/reflection \
--train_batch_size 1 \
--learning_rate 1e-05 \
--learning_rate_attn 1e-05 \
--resolution 512 \
--num_train_epochs 1000 \
--checkpointing_steps 1000 \
--color_augmentation \
--flip_augmentation \
--seed 42 \
--checkpoints_total_limit 10 \
--random_mask_dilation \
--random_mask_erosion \
--lr_scheduler cosine \
--background_loss_weight 1 \
--gradient_accumulation_steps 4 \
--object_localization \
--object_localization_weight 0.01 \
# --pretrained_path ./exp/3.27_train/checkpoint-1000/model.safetensors \
# --pretrained_path_postfuse ./exp/3.27_train/checkpoint-1000/model_1.safetensors \