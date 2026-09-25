import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_kmeans import CosineSimilarity, KMeans
from typing import List
import HYPIR.model.dvt.models as DVT

from .annotation import add_label, draw_label
from .layout import add_border, hcat, vcat


def get_robust_pca(features: Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    features = features.float()
    print(f"dtype of features {features.dtype}")
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(feat_map, img_size, interp="nearest", return_pca_stats=False, pca_stats=None):
    """
    Computes PCA visualization matching the style of visualization.py:
    1. Standardization (Zero Mean, Unit Variance) - Critical for good contrast
    2. PCA (Top 3 components)
    3. Min-Max Normalization to [0, 1]
    """
    # Ensure feat_map is (1, H, W, C)
    feat_map = feat_map[None] if feat_map.dim() == 3 else feat_map
    
    # Flatten features: (N, C)
    N, C = feat_map.numel() // feat_map.shape[-1], feat_map.shape[-1]
    flat_feat = feat_map.reshape(N, C).float()

    if pca_stats is None:
        # 1. Standardization: (X - Mean) / Std
        # add epsilon to avoid div by zero
        mean = flat_feat.mean(dim=0, keepdim=True)
        std = flat_feat.std(dim=0, keepdim=True) + 1e-8
        flat_feat_std = (flat_feat - mean) / std

        # 2. PCA Calculation
        # Disable autocast to prevent BFloat16/Half errors in SVD
        with torch.autocast("cuda", enabled=False):
            # torch.pca_lowrank returns U, S, V. We need V (projection matrix).
            _, _, V = torch.pca_lowrank(flat_feat_std, q=3, niter=20)
        
        reduct_mat = V
        norm_mean = mean
        norm_std = std
    else:
        reduct_mat, norm_mean, norm_std = pca_stats
    # Apply Transform
    flat_feat_std = (flat_feat - norm_mean) / norm_std
    pca_projected = flat_feat_std @ reduct_mat

    # 3. Min-Max Normalization to [0, 1]
    # Calculate min/max across the spatial dimensions but independently for each RGB channel
    pca_min = pca_projected.min(dim=0, keepdim=True).values
    pca_max = pca_projected.max(dim=0, keepdim=True).values
    pca_color = (pca_projected - pca_min) / (pca_max - pca_min + 1e-8)
    
    # Reshape back to spatial dimensions
    pca_color = pca_color.reshape(*feat_map.shape[:-1], 3) # (1, H, W, 3)
    
    # Interpolate to target image size
    pca_color = pca_color.permute(0, 3, 1, 2) # (1, 3, H, W) for grid_sample/interpolate
    pca_color = F.interpolate(pca_color, size=img_size, mode=interp)
    pca_color = pca_color.permute(0, 2, 3, 1).cpu().numpy().squeeze(0) # (H_out, W_out, 3)

    if return_pca_stats:
        return pca_color, (reduct_mat, norm_mean, norm_std)
    return pca_color



def get_scale_map(scalar_map, img_size, interp="nearest"):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    scalar_map = torch.norm(scalar_map, dim=-1, keepdim=True)
    if scalar_map.shape[0] != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min()) / (scalar_map.max() - scalar_map.min() + 1e-6)
    scalar_map = F.interpolate(scalar_map.permute(0, 3, 1, 2), size=img_size, mode=interp)
    scalar_map = scalar_map.permute(0, 2, 3, 1).squeeze(-1)
    cmap = plt.get_cmap("inferno")
    scalar_map = cmap(scalar_map.float().cpu().numpy().squeeze(0))[..., :3]
    return scalar_map


def get_similarity_map(features: Tensor, img_size=(224, 224)):
    """
    compute the similarity map of the central patch to the rest of the image
    """
    assert len(features.shape) == 4, "features should be (1, C, H, W)"
    H, W, C = features.shape[1:]
    center_patch_feature = features[0, H // 2, W // 2, :]
    center_patch_feature_normalized = center_patch_feature / center_patch_feature.norm()
    center_patch_feature_normalized = center_patch_feature_normalized.unsqueeze(1)
    # Reshape and normalize the entire feature tensor
    features_flat = features.view(-1, C)
    features_normalized = features_flat / features_flat.norm(dim=1, keepdim=True)

    similarity_map_flat = features_normalized @ center_patch_feature_normalized
    # Reshape the flat similarity map back to the spatial dimensions (H, W)
    similarity_map = similarity_map_flat.view(1, 1, H, W)
    sim_min, sim_max = similarity_map.min(), similarity_map.max()
    similarity_map = (similarity_map - sim_min) / (sim_max - sim_min)

    # we don't want the center patch to be the most similar
    similarity_map[0, 0, H // 2, W // 2] = -1.0
    similarity_map = F.interpolate(similarity_map, size=img_size, mode="bilinear")
    similarity_map = similarity_map.squeeze(0).squeeze(0)

    similarity_map_np = similarity_map.float().cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb


def get_cluster_map(feat_map, img_size, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, distance=CosineSimilarity, verbose=False)
    if feat_map.shape[0] != 1:
        feat_map = feat_map[None]  # make it (1, h, w, C)
    labels = kmeans.fit_predict(feat_map.reshape(1, -1, feat_map.shape[-1])).float()
    labels = F.interpolate(labels.reshape(1, *feat_map.shape[:-1]), size=img_size, mode="nearest")
    labels = labels.squeeze().cpu().numpy().astype(int)
    cmap = plt.get_cmap("rainbow", num_clusters)
    cluster_map = cmap(labels)[..., :3]
    return cluster_map.reshape(img_size[0], img_size[1], 3)


def visualize_offline_denoised_samples(
    denoiser: DVT.SingleImageDenoiser,
    neural_field: DVT.NeuralFeatureField,
    raw_features: torch.Tensor,
    coord: torch.Tensor,
    patch_images: torch.Tensor,
    device: torch.device = torch.device("cuda"),
    denormalizer=None,
    dtype=torch.float32,
):
    pca_samples = []
    for i in range(len(raw_features)):
        data_dict = {
            "transformed_view": patch_images[i : i + 1].to(device),
            "pixel_coords": coord[i : i + 1].to(device),
        }
        img = data_dict["transformed_view"]
        hw = img.shape[-2:]
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = denoiser.forward(
                raw_vit_outputs=raw_features[i : i + 1],
                global_pixel_coords=data_dict["pixel_coords"],
                neural_field=neural_field,
                return_visualization=True,
            )
        # gt noisy features
        gt_raw_features = output["raw_vit_outputs"].float()
        shared_patterns = output["shared_patterns"].float()
        denoised_feats = output["denoised_feats"].float()

        # compute the similarity of the central patch to the rest of the image
        gt_feature_pca = get_pca_map(gt_raw_features, hw)
        gt_cluster_map = get_cluster_map(gt_raw_features, hw, num_clusters=5)
        gt_norm_map = get_scale_map(gt_raw_features, hw)
        gt_similarity_map = get_similarity_map(gt_raw_features, hw)

        # shared artifact: G in the paper
        shared_artifact_pca = get_pca_map(shared_patterns, hw)
        # noise_norm = get_scale_map(shared_patterns, hw)

        # denoised_feats from neural fields: F in the paper
        denoised_pca = get_pca_map(denoised_feats, hw)
        denoised_cluster_map = get_cluster_map(denoised_feats, hw, num_clusters=5)
        denoised_norm_map = get_scale_map(denoised_feats, hw)
        denoised_similarity_map = get_similarity_map(denoised_feats, hw)

        # real residual features
        # gt_residual_features = gt_raw_features - denoised_feats - shared_patterns
        # gt_residual_norm = get_scale_map(gt_residual_features, hw)

        # undo standardization
        img = denormalizer(img)
        img = img.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        pca_sample = [
            img,
            gt_feature_pca,
            gt_cluster_map,
            gt_norm_map,
            gt_similarity_map,
            denoised_pca,
            denoised_cluster_map,
            denoised_norm_map,
            denoised_similarity_map,
            shared_artifact_pca,
            # noise_norm,
            # gt_residual_norm,
        ]
        if "pred_residual" in output:  # h in the paper
            # the norm of residual features
            pred_residual_norm = get_scale_map(output["pred_residual"], hw)
            # combine shared artifact and residual features
            shared_patterns_and_residual = output["shared_patterns_and_residual"].float()
            shared_patterns_and_residual_color = get_pca_map(shared_patterns_and_residual, hw)
            # the norm of full residual features
            pca_sample.append(pred_residual_norm)
            pca_sample.append(shared_patterns_and_residual_color)
        pca_sample = [torch.tensor(sample).permute(2, 0, 1) for sample in pca_sample]
        if i == 0:
            pca_sample = hcat(
                add_label(pca_sample[0], "Input Image", font_size=58),
                add_label(pca_sample[1], "Original Feature", font_size=58),
                add_label(pca_sample[2], "Original Cluster", font_size=58),
                add_label(pca_sample[3], "Original Norm", font_size=58),
                add_label(pca_sample[4], "Original Sim", font_size=58),
                add_label(pca_sample[5], "Denoised Feat (F)", font_size=58),
                add_label(pca_sample[6], "Denoised Cluster", font_size=58),
                add_label(pca_sample[7], "Denoised Norm", font_size=58),
                add_label(pca_sample[8], "Denoised Sim", font_size=58),
                add_label(pca_sample[9], "Shared Noise (G)", font_size=58),
                add_label(pca_sample[10], "Residual Norm (h)", font_size=58),
                add_label(pca_sample[11], "Composited (G+h)", font_size=58),
                gap=12,
            )
        else:
            pca_sample = hcat(*pca_sample, gap=12)
        pca_samples.append(pca_sample)
    pca_samples = add_border(vcat(*pca_samples))
    pca_samples = pca_samples.permute(1, 2, 0).cpu().numpy()
    pca_samples = (pca_samples * 255).astype(np.uint8)
    return pca_samples, output["denoised_feats"].detach().float().cpu().numpy()


def visualize_online_denoised_samples(
    data_dict: dict,
    latents_dict: dict,
    pred_denoised_feats: torch.Tensor,
    denormalizer=None,
    num_samples: int = 5,
):
    hw = data_dict["GT"].shape[-2:]
    pca_samples = []
    for i in range(num_samples):
        image = data_dict["GT"][i].cpu().permute(1, 2, 0).numpy()
        origin_img = latents_dict["origin_img"][i].cpu().permute(1, 2, 0).float().numpy()
        gt_feats = latents_dict["z_hq"][i].permute(1, 2, 0).float()
        gt_pca = get_pca_map(gt_feats, hw)
        gt_norm = get_scale_map(gt_feats, hw)
        pred_feats = pred_denoised_feats[i].permute(1, 2, 0).float()
        pred_denoised_color = get_pca_map(pred_feats, hw)
        pred_denoised_norm_color = get_scale_map(pred_feats, hw)
        pca_sample = [
            image,
            origin_img,
            gt_pca,
            gt_norm,
            pred_denoised_color,
            pred_denoised_norm_color,
        ]
        pca_sample = [torch.tensor(sample).permute(2, 0, 1) for sample in pca_sample]
        if i == 0:
            pca_sample = hcat(
                add_label(pca_sample[0], "Input Image", font_size=58),
                add_label(pca_sample[1], "Origin Output", font_size=58),
                add_label(pca_sample[2], "GT Feature", font_size=58),
                add_label(pca_sample[3], "GT Norm", font_size=58),
                add_label(pca_sample[4], "Pred Denoised", font_size=58),
                add_label(pca_sample[5], "Pred Denoised Norm", font_size=58),
                gap=12,
            )
        else:
            pca_sample = hcat(*pca_sample, gap=12)
        pca_samples.append(pca_sample)

    pca_samples = add_border(vcat(*pca_samples))
    pca_samples = pca_samples.permute(1, 2, 0).cpu().numpy()
    pca_samples = (pca_samples * 255).astype(np.uint8)
    return pca_samples


def visualize_samples(
    image: torch.Tensor,
    # gt_feature: torch.Tensor,
    features: List[torch.Tensor],
    denormalizer=None,
    sample_idx=[29],
    save_path: str | None = None
):
    """
    Args:
        image: CHW or BCHW tensor.
        features: list of feature maps shaped (H, W, C) or (1, H, W, C).
        denormalizer: callable to undo image normalization.
        num_samples: number of feature items to visualize.
        save_path: optional path to save the composed panel.
    Returns:
        np.ndarray uint8 panel laid out in one row.
    """
    # 1. Prepare Input Image
    if image.dim() == 4:
        image = image[0]
    
    # img_tensor = image if denormalizer is None else denormalizer(image[None]).squeeze(0)
    # img_np = img_tensor.permute(1, 2, 0).float().cpu().numpy()
    
    # # Store components to be visualized as (numpy_image_hwc, label_string)
    # vis_items = [(img_np, "Input")]
    vis_items = []
    # 2. Process Selected Features
    # gt_feat = gt_feature.float()
    # gt_pca = get_pca_map(gt_feat, (512, 512))
    # gt_norm = get_scale_map(gt_feat, (512, 512))
    # gt_cluster = get_cluster_map(gt_feat, (512, 512), num_clusters=5)
    # gt_similarity = get_similarity_map(gt_feat, (512, 512))
    
    # vis_items.append((gt_pca, f"GT PCA"))
    # vis_items.append((gt_norm, f"GT Norm"))
    # vis_items.append((gt_cluster, f"GT Cluster"))
    # vis_items.append((gt_similarity, f"GT Similarity"))

    for idx in sample_idx:
        feat = features[idx]
        
        # Handle flattened features (B, L, C) -> (B, H, W, C) assuming square shape
        if feat.dim() == 3:
            B, L, C = feat.shape
            h = w = int(L**0.5)
            if h * w == L:
                feat = feat.view(B, h, w, C)
        
        feat = feat.float()
        hw = (512, 512)
        
        pca = get_pca_map(feat, hw)
        norm = get_scale_map(feat, hw)
        cluster = get_cluster_map(feat, hw, num_clusters=5)
        similarity = get_similarity_map(feat, hw)
        
        vis_items.append((pca, f"Layer{idx} PCA"))
        vis_items.append((norm, f"Layer{idx} Norm"))
        vis_items.append((cluster, f"Layer{idx} Cluster"))
        vis_items.append((similarity, f"Layer{idx} Similarity"))

    # 3. Layout: Convert to Tensor, Add Labels, Concatenate
    labeled_tensors = []
    for comp_np, label_text in vis_items:
        # Convert numpy (H, W, C) -> Tensor (C, H, W)
        comp_t = torch.from_numpy(comp_np).permute(2, 0, 1).float()
        labeled_tensors.append(add_label(comp_t, label_text, font_size=48))

    panel = add_border(hcat(*labeled_tensors, gap=12))
    panel_np = (panel.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    if save_path is not None:
        plt.imsave(save_path, panel_np)
        
    return panel_np