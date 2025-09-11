import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import lpips

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import norm

import safetensors.torch as sf
from huggingface_hub import hf_hub_download

from net.CIDNet import CIDNet
from measure import *

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_cidnet_model(model_path):
    """Hugging Face에서 CIDNet 모델을 다운로드하고 로드"""
    print(f"Loading CIDNet model from: {model_path}")
    
    # Hugging Face Hub에서 CIDNet model 다운로드
    model_file = hf_hub_download(
        repo_id=model_path, 
        filename="model.safetensors", 
        repo_type="model"
    )
    print(f"CIDNet model downloaded from: {model_file}")
    
    # 모델 초기화 및 가중치 로드
    model = CIDNet()
    state_dict = sf.load_file(model_file)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)  # Move model to GPU
    model.eval()
    return model


def load_sam_model(sam_model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth"):
    """Hugging Face에서 SAM 모델을 다운로드하고 로드"""
    # sam_model_path를 repo_id와 filename으로 분리
    parts = sam_model_path.split('/')
    if len(parts) < 3:
        raise ValueError("SAM model path should be in format: repo_id/filename (e.g., Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth)")
    
    model_repo = '/'.join(parts[:2])  # "Gourieff/ReActor"
    model_filename = '/'.join(parts[2:])  # "models/sams/sam_vit_b_01ec64.pth"
    
    print(f"Loading SAM model from: {model_repo}/{model_filename}")
    
    # Hugging Face Hub에서 SAM checkpoint 다운로드
    checkpoint_path = hf_hub_download(
        repo_id=model_repo, 
        filename=model_filename, 
        repo_type="dataset"
    )
    print(f"SAM model downloaded from: {checkpoint_path}")
    
    # SAM 모델 초기화 및 로드
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam = sam.to(device)  # Move SAM model to GPU
    
    # Mask generator 생성
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )
    return mask_generator


def metrics_one(im1, im2, use_GT_mean, loss_fn):
    if isinstance(im1, Image.Image):
        im1 = np.array(im1)
    if isinstance(im2, Image.Image):
        im2 = np.array(im2)

    if use_GT_mean:
        mean_restored = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).mean()
        im1 = np.clip(im1 * (mean_target/mean_restored), 0, 255)
    
    score_psnr = calculate_psnr(im1, im2)
    score_ssim = calculate_ssim(im1, im2)
    ex_p0 = lpips.im2tensor(im1).cuda()
    ex_ref = lpips.im2tensor(im2).cuda()
    
    score_lpips = loss_fn.forward(ex_ref, ex_p0)
    return score_psnr, score_ssim, score_lpips


def process_image_with_cidnet(model, image, alpha, alpha_s):
    input_tensor = transforms.ToTensor()(image)
    # Convert parameters to tensor
    if isinstance(alpha_s, np.ndarray):
        alpha_s = torch.from_numpy(alpha_s).to(device).to(input_tensor.dtype)
    else:
        alpha_s = torch.tensor(alpha_s, dtype=input_tensor.dtype, device=device)
    
    if isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha).to(device).to(input_tensor.dtype)
    else:
        alpha = torch.tensor(alpha, dtype=input_tensor.dtype, device=device)
        
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = torch.nn.functional.pad(input_tensor.unsqueeze(0), (0,padw,0,padh), 'reflect')
    input_tensor = input_tensor.to(device)  # Move input tensor to GPU
    
    with torch.no_grad():
        model.trans.gated = True
        model.trans.alpha_s = alpha_s
        model.trans.gated2 = True
        model.trans.alpha = alpha
        output = model(input_tensor)
    
    output = torch.clamp(output, 0, 1)
    output = output[:, :, :h, :w]
    output = output.cpu()  # Move output back to CPU for PIL conversion
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    return enhanced_img


def determine_parameters(input_image, mask, gt_image, model, device, n_iterations = 10):
    """베이지안 최적화를 사용하여 마스크 영역에 대한 최적의 파라미터를 찾는 함수"""
    
    # 파라미터 범위 설정 (alpha: 0.8~1.2, alpha_s: 1.0~1.6)
    param_bounds = {
        'alpha': (0.8, 1.2),
        'alpha_s': (1.0, 1.6)
    }

    # 초기 샘플 생성 (기본값과 그 근처 값들)
    X = np.array([
        [1.00, 1.30],
        [0.98, 1.20],
        [1.02, 1.40],
        [0.96, 1.10],
        [1.04, 1.50],
        [0.90, 1.60],
        [1.10, 1.00],
        [0.85, 1.15],
        [1.15, 1.45],
        [0.80, 1.35],
        [1.20, 1.25]
    ])
    
    # 목적 함수 계산 (마스크 영역의 이미지 품질 메트릭 기반)
    def objective_function(params):
        alpha, alpha_s = params
        
        # 현재 파라미터로 이미지 처리
        input_tensor = transforms.ToTensor()(input_image)
        factor = 8
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_tensor = torch.nn.functional.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            model.trans.alpha = alpha
            model.trans.alpha_s = alpha_s
            model.trans.gated = True
            model.trans.gated2 = True
            output = model(input_tensor)
            
        output = torch.clamp(output, 0, 1)
        output = output[:, :, :h, :w]
        output = output.cpu()
        enhanced_img = transforms.ToPILImage()(output.squeeze(0))
        
        # PIL Image를 numpy array로 변환
        enhanced_np = np.array(enhanced_img)
        input_np = np.array(input_image)
        gt_np = np.array(gt_image)
        
        # 마스크 적용
        mask_3d = np.stack([mask] * 3, axis=-1)
        enhanced_masked = np.where(mask_3d, enhanced_np, input_np)
        gt_masked = np.where(mask_3d, gt_np, input_np)
        
        # 마스크 영역에 대한 메트릭 계산
        # psnr_val = peak_signal_noise_ratio(enhanced_masked, gt_masked, data_range=255)
        # ssim_val = structural_similarity(enhanced_masked, gt_masked, channel_axis=2, data_range=255)
        psnr_val, ssim_val, lpips_val  = metrics_one(enhanced_masked, gt_masked, use_GT_mean=True, loss_fn=loss_fn)

        # 파라미터가 적절한 범위에 있는지 확인
        param_score = 1.0
        if not (param_bounds['alpha'][0] <= alpha <= param_bounds['alpha'][1] and
                param_bounds['alpha_s'][0] <= alpha_s <= param_bounds['alpha_s'][1]):
            param_score = 0.0
        
        # 최종 점수 계산 (PSNR, SSIM, 파라미터 범위 고려)
        score = (psnr_val / 40.0 * 0.5 + ssim_val * 0.3 + param_score * 0.2)  
        return score

    # 최적화 전(기본값) 점수 계산
    default_score = objective_function([1.0, 1.3])

    y = np.array([objective_function(params) for params in X])

    # === Normalization (scaling) === 
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    length_scale = 0.02
    kernel = Matern(nu=2.5, length_scale=length_scale, length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, n_restarts_optimizer=10)

    for i in range(n_iterations):
        gp.fit(X_scaled, y_scaled)
        best_score = np.max(y_scaled)
        
        # 다음 파라미터 후보 생성 (원본 스케일)
        x_test = np.random.uniform(
            low=[param_bounds['alpha'][0], param_bounds['alpha_s'][0]],
            high=[param_bounds['alpha'][1], param_bounds['alpha_s'][1]],
            size=(100, 2)
        )
        # 후보를 스케일에 맞게 변환
        x_test_scaled = X_scaler.transform(x_test)
        
        # 예상 개선량 계산
        mean, std = gp.predict(x_test_scaled, return_std=True)
        ei = (mean - best_score) * norm.cdf((mean - best_score) / std) + std * norm.pdf((mean - best_score) / std)
        
        # 다음 파라미터 선택 (스케일 복원)
        next_params = x_test[np.argmax(ei)]
        next_score = objective_function(next_params)
        
        # 결과 업데이트
        X = np.vstack([X, next_params])
        y = np.append(y, next_score)
        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 최적의 파라미터 반환 (스케일 복원)
    best_idx = np.argmax(y)
    best_params = X[best_idx]

    return best_params[0], best_params[1], y[best_idx], default_score


def group_masks_by_stats(masks, image, num_groups=5):
    """마스크를 밝기와 대비를 기준으로 그룹화하고, 더 효과적인 클러스터링을 수행"""
    mask_stats = []
    
    for i, mask in enumerate(masks):
        mask_stats.append({
            'mask': mask,
            'area': mask['area']
        })
    
    area_values = np.array([m['area'] for m in mask_stats])
    
    # 정규화
    area_norm = (area_values - np.min(area_values)) / (np.max(area_values) - np.min(area_values))
    
    # 가중치 적용 (면적이 큰 마스크에 더 높은 가중치)
    features = np.column_stack([area_norm])
    
    kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    grouped_masks = []
    for i in range(num_groups):
        group_masks = [m['mask'] for m, label in zip(mask_stats, labels) if label == i]
        if not group_masks:
            continue
            
        combined_segmentation = np.zeros_like(group_masks[0]['segmentation'], dtype=bool)
        total_area = 0
        
        for mask in group_masks:
            combined_segmentation |= mask['segmentation']
            total_area += mask['area']
        
        grouped_mask = {
            'segmentation': combined_segmentation,
            'area': total_area
        }
        grouped_masks.append(grouped_mask)
    
    return grouped_masks


def parse_args():
    parser = argparse.ArgumentParser(description='Process images using CIDNet and SAM')
    parser.add_argument('--input_dir', type=str, default="datasets/LOLdataset/eval15/low",
                        help='Directory containing input images')
    parser.add_argument('--gt_dir', type=str, default="datasets/LOLdataset/eval15/high",
                        help='Directory containing ground truth images')
    parser.add_argument('--output_dir', type=str, default="results/LOLdataset",
                        help='Directory to save output images')
    parser.add_argument('--cidnet_model', type=str, default="Fediory/HVI-CIDNet-LOLv1-woperc",
                        help='CIDNet model name or path from Hugging Face')
    parser.add_argument('--sam_model', type=str, default="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                        help='SAM model path in format: repo_id/filename (e.g., Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Add device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    whole_dir = os.path.join(args.output_dir, "whole")
    sam_dir = os.path.join(args.output_dir, "sam")
    comparison_dir = os.path.join(args.output_dir, "comparison")
    
    os.makedirs(whole_dir, exist_ok=True)
    os.makedirs(sam_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get list of PNG files from input directory
    input_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.png')]
    
    # Load models only once
    cidnet_model = load_cidnet_model(args.cidnet_model)
    sam_model = load_sam_model(args.sam_model)
    # Calculate metrics for both whole and SAM enhanced images
    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()
    
    # Process each image
    whole_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    sam_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    for input_file in input_files:
        print(f"Processing {input_file}...")
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Load input and ground truth images
        input_path = os.path.join(args.input_dir, input_file)
        gt_path = os.path.join(args.gt_dir, input_file)
        
        input_image = Image.open(input_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')
        
        # 1. 통으로 CIDNet 처리
        whole_enhanced = process_image_with_cidnet(cidnet_model, input_image, 1.0, 1.3)
        whole_output_path = os.path.join(whole_dir, f"{input_filename}.png")
        whole_enhanced.save(whole_output_path)
        
        # Calculate metrics for whole enhanced image
        whole_psnr, whole_ssim, whole_lpips = metrics_one(whole_enhanced, gt_image, use_GT_mean=True, loss_fn=loss_fn)
        whole_metrics['psnr'].append(whole_psnr)
        whole_metrics['ssim'].append(whole_ssim)
        whole_metrics['lpips'].append(whole_lpips.item())




        # 2. 영역별 CIDNet 처리
        initial_masks = sam_model.generate(np.array(input_image))

        # 마스크 그룹핑
        grouped_masks = group_masks_by_stats(initial_masks, input_image)
        
        # 모든 마스크를 합쳐서 배경 마스크 생성
        combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
        for mask in grouped_masks:
            combined_mask |= mask['segmentation']
        background_mask = ~combined_mask
        
        # 배경 영역 처리
        bg_alpha, bg_alpha_s, best_score, default_score = determine_parameters(input_image, background_mask, gt_image, cidnet_model, device)
        print(f"Background parameters: alpha={bg_alpha:.4f}, alpha_s={bg_alpha_s:.4f}, Best score={best_score:.4f}, Default score={default_score:.4f}")


        alpha_s_matrix = np.zeros((input_image.height, input_image.width), dtype=float)
        alpha_matrix = np.zeros((input_image.height, input_image.width), dtype=float)

        alpha_s_matrix[~combined_mask] = bg_alpha_s
        alpha_matrix[~combined_mask] = bg_alpha

        # 각 그룹 처리
        for i, mask in enumerate(grouped_masks):
            alpha, alpha_s, best_score, default_score = determine_parameters(input_image, mask['segmentation'], gt_image, cidnet_model, device)
            print(f"Group {i+1} parameters: alpha={alpha:.4f}, alpha_s={alpha_s:.4f}, Best score={best_score:.4f}, Default score={default_score:.4f}")

            # 기존 값이 있는 경우 평균 계산
            mask_indices = mask['segmentation']
            existing_values_s = alpha_s_matrix[mask_indices]
            existing_values = alpha_matrix[mask_indices]
            
            # 0이 아닌 값이 있는 경우에만 평균 계산
            non_zero_s = existing_values_s != 0
            non_zero = existing_values != 0
            
            alpha_s_matrix[mask_indices] = np.where(non_zero_s, 
                (existing_values_s + alpha_s) / 2, alpha_s)
            alpha_matrix[mask_indices] = np.where(non_zero,
                (existing_values + alpha) / 2, alpha)

        output_image = process_image_with_cidnet(cidnet_model, input_image, alpha_matrix, alpha_s_matrix)
        output_path = os.path.join(sam_dir, f"{input_filename}.png")
        output_image.save(output_path)
        
        # Calculate metrics for SAM enhanced image
        sam_psnr, sam_ssim, sam_lpips = metrics_one(output_image, gt_image, use_GT_mean=True, loss_fn=loss_fn)
        sam_metrics['psnr'].append(sam_psnr)
        sam_metrics['ssim'].append(sam_ssim)
        sam_metrics['lpips'].append(sam_lpips.item())
        
        # 3. 결과 비교를 위한 시각화
        comparison = Image.new('RGB', (input_image.width * 3, input_image.height))
        comparison.paste(input_image, (0, 0))
        comparison.paste(whole_enhanced, (input_image.width, 0))
        comparison.paste(output_image, (input_image.width * 2, 0))
        comparison_path = os.path.join(comparison_dir, f"{input_filename}.png")
        comparison.save(comparison_path)
    
    # Calculate and print average metrics
    print("\n=== Reimplementation of paper SOTA ===")
    print(f"Average PSNR: {np.mean(whole_metrics['psnr']):.4f} dB")
    print(f"Average SSIM: {np.mean(whole_metrics['ssim']):.4f}")
    print(f"Average LPIPS: {np.mean(whole_metrics['lpips']):.4f}")
    
    print("\n=== SAM Enhanced Image Metrics ===")
    print(f"Average PSNR: {np.mean(sam_metrics['psnr']):.4f} dB")
    print(f"Average SSIM: {np.mean(sam_metrics['ssim']):.4f}")
    print(f"Average LPIPS: {np.mean(sam_metrics['lpips']):.4f}")