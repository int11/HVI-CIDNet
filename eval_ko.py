#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 지원 평가 스크립트
Korean Language Support Evaluation Script
"""
import os
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.CIDNet import CIDNet
from measure import metrics
import dist

# 기존 eval 함수를 import
from eval import eval

if __name__ == '__main__':
    eval_parser = argparse.ArgumentParser(
        description='HVI-CIDNet 평가 도구 / HVI-CIDNet Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시 / Usage Examples:
    # LOLv1 데이터셋 평가
    python eval_ko.py --lol
    
    # LOLv2-real 데이터셋 평가 
    python eval_ko.py --lol_v2_real --best_GT_mean
    
    # 언페어드 데이터셋 평가 (DICM)
    python eval_ko.py --unpaired --DICM --unpaired_weights ./weights/LOLv2_syn/w_perc.pth --alpha 0.9
        """)
    
    # 기본 옵션
    eval_parser.add_argument('--perc', action='store_true', 
                           help='인식 손실로 학습된 모델 / trained with perceptual loss')
    
    # 페어드 데이터셋 옵션
    paired_group = eval_parser.add_argument_group('페어드 데이터셋 / Paired Datasets')
    paired_group.add_argument('--lol', action='store_true', 
                            help='LOLv1 데이터셋 출력 / output lolv1 dataset', default=True)
    paired_group.add_argument('--lol_v2_real', action='store_true', 
                            help='LOL-v2-real 데이터셋 출력 / output lol_v2_real dataset')
    paired_group.add_argument('--lol_v2_syn', action='store_true', 
                            help='LOL-v2-syn 데이터셋 출력 / output lol_v2_syn dataset')
    paired_group.add_argument('--SICE_grad', action='store_true', 
                            help='SICE-grad 데이터셋 출력 / output SICE_grad dataset')
    paired_group.add_argument('--SICE_mix', action='store_true', 
                            help='SICE-mix 데이터셋 출력 / output SICE_mix dataset')
    
    # LOL-v2-real 최적화 옵션
    lol_v2_group = eval_parser.add_argument_group('LOL-v2-real 최적화 옵션 / LOL-v2-real Optimization Options')
    lol_v2_group.add_argument('--best_GT_mean', action='store_true', 
                            help='LOL-v2-real 최고 GT 평균 / lol_v2_real dataset best_GT_mean')
    lol_v2_group.add_argument('--best_PSNR', action='store_true', 
                            help='LOL-v2-real 최고 PSNR / lol_v2_real dataset best_PSNR')
    lol_v2_group.add_argument('--best_SSIM', action='store_true', 
                            help='LOL-v2-real 최고 SSIM / lol_v2_real dataset best_SSIM')
    
    # 언페어드 데이터셋 옵션
    unpaired_group = eval_parser.add_argument_group('언페어드 데이터셋 / Unpaired Datasets')
    unpaired_group.add_argument('--unpaired', action='store_true', 
                              help='언페어드 데이터셋 출력 / output unpaired dataset')
    unpaired_group.add_argument('--DICM', action='store_true', 
                              help='DICM 데이터셋 출력 / output DICM dataset')
    unpaired_group.add_argument('--LIME', action='store_true', 
                              help='LIME 데이터셋 출력 / output LIME dataset')
    unpaired_group.add_argument('--MEF', action='store_true', 
                              help='MEF 데이터셋 출력 / output MEF dataset')
    unpaired_group.add_argument('--NPE', action='store_true', 
                              help='NPE 데이터셋 출력 / output NPE dataset')
    unpaired_group.add_argument('--VV', action='store_true', 
                              help='VV 데이터셋 출력 / output VV dataset')
    
    # 커스텀 데이터셋 옵션
    custom_group = eval_parser.add_argument_group('커스텀 데이터셋 / Custom Dataset')
    custom_group.add_argument('--custome', action='store_true', 
                            help='커스텀 데이터셋 출력 / output custom dataset')
    custom_group.add_argument('--custome_path', type=str, default='./YOLO',
                            help='커스텀 데이터셋 경로 / custom dataset path')
    
    # 파라미터 옵션
    param_group = eval_parser.add_argument_group('파라미터 / Parameters')
    param_group.add_argument('--alpha', type=float, default=1.0,
                           help='알파 값 (조명 스케일) / alpha value (illumination scale)')
    param_group.add_argument('--unpaired_weights', type=str, 
                           default='./weights/LOLv2_syn/w_perc.pth',
                           help='언페어드 가중치 경로 / unpaired weights path')

    ep = eval_parser.parse_args()

    # GPU 확인
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("GPU를 찾을 수 없거나 CUDA_VISIBLE_DEVICES 번호를 변경해야 합니다 / No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    # 출력 디렉토리 생성
    if not os.path.exists('./output'):          
        os.mkdir('./output')  
    
    print("=== HVI-CIDNet 한국어 평가 도구 ===")
    print("=== HVI-CIDNet Korean Evaluation Tool ===")
    print()
    
    # 원본 eval.py의 나머지 코드 실행
    from eval import *  # 기존 eval.py의 모든 기능 import