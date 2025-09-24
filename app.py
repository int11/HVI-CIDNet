import numpy as np
import torch
import gradio as gr
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import imquality.brisque as brisque
from loss.niqe_utils import *
import platform
import argparse

# 한국어 지원을 위한 언어 설정
TRANSLATIONS = {
    'en': {
        'title': "HVI-CIDNet (Low-Light Image Enhancement)",
        'input_image': "Low-light Image",
        'image_score': "Image Score",
        'image_score_info': "Calculate NIQE and BRISQUE, default is \"No\".",
        'model_weights': "Model Weights",
        'model_weights_info': "Choose your model. The best models are \"SICE.pth\" and \"generalization.pth\".",
        'gamma_curve': "gamma curve",
        'gamma_curve_info': "Lower is lighter, and best range is [0.5,2.5].",
        'alpha_s': "Alpha-s",
        'alpha_s_info': "Higher is more saturated.",
        'alpha_i': "Alpha-i", 
        'alpha_i_info': "Higher is lighter.",
        'result': "Result",
        'niqe': "NIQE",
        'niqe_info': "Lower is better.",
        'brisque': "BRISQUE",
        'brisque_info': "Lower is better."
    },
    'ko': {
        'title': "HVI-CIDNet (저조도 이미지 향상)",
        'input_image': "저조도 이미지",
        'image_score': "이미지 점수",
        'image_score_info': "NIQE와 BRISQUE 점수 계산, 기본값은 \"No\"입니다.",
        'model_weights': "모델 가중치",
        'model_weights_info': "모델을 선택하세요. 최고의 모델은 \"SICE.pth\"와 \"generalization.pth\"입니다.",
        'gamma_curve': "감마 곡선",
        'gamma_curve_info': "값이 낮을수록 더 밝아집니다. 최적 범위는 [0.5,2.5]입니다.",
        'alpha_s': "알파-s",
        'alpha_s_info': "값이 클수록 채도가 높아집니다.",
        'alpha_i': "알파-i",
        'alpha_i_info': "값이 클수록 더 밝아집니다.",
        'result': "결과",
        'niqe': "NIQE",
        'niqe_info': "낮을수록 좋습니다.",
        'brisque': "BRISQUE", 
        'brisque_info': "낮을수록 좋습니다."
    }
}

opt_parser = argparse.ArgumentParser(description='App')
opt_parser.add_argument('--cpu', action='store_true', help='CPU-Only')
opt_parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko'], 
                        help='언어 선택 / Language selection (en/ko)')
opt = opt_parser.parse_args()

# 선택된 언어의 번역 텍스트 가져오기
lang = opt.lang
texts = TRANSLATIONS[lang]

if opt.cpu:
    eval_net = CIDNet().cpu()
else:
    eval_net = CIDNet().cuda()
    
eval_net.trans.gated = True
eval_net.trans.gated2 = True

def process_image(input_img,score,model_path,gamma,alpha_s=1.0,alpha_i=1.0):
    torch.set_grad_enabled(False)
    eval_net.load_state_dict(torch.load(os.path.join(directory,model_path), map_location=lambda storage, loc: storage))
    eval_net.eval()
    
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect')
    with torch.no_grad():
        eval_net.trans.alpha_s = alpha_s
        eval_net.trans.alpha = alpha_i
        if opt.cpu:
            output = eval_net(input**gamma)
        else:
            output = eval_net(input.cuda()**gamma)
            
    if opt.cpu:
        output = torch.clamp(output,0,1)
    else:
        output = torch.clamp(output.cuda(),0,1).cuda()
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    if score == 'Yes':
        im1 = enhanced_img.convert('RGB')
        score_brisque = brisque.score(im1) 
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
        return enhanced_img,score_niqe,score_brisque
    else:
        return enhanced_img,0,0

def find_pth_files(directory):
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if 'train' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files

def remove_weights_prefix(paths):
    os_name = platform.system()
    if os_name.lower() == 'windows':
        cleaned_paths = [path.replace('weights\\', '') for path in paths]
    elif os_name.lower() == 'linux':
        cleaned_paths = [path.replace('weights/', '') for path in paths]
        
    return cleaned_paths

directory = "weights"
pth_files = find_pth_files(directory)
pth_files2 = remove_weights_prefix(pth_files)

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label=texts['input_image'], type="pil"),
        gr.Radio(choices=['Yes','No'], label=texts['image_score'], info=texts['image_score_info']),
        gr.Radio(choices=pth_files2, label=texts['model_weights'], info=texts['model_weights_info']),
        gr.Slider(0.1,5, label=texts['gamma_curve'], step=0.01, value=1.0, info=texts['gamma_curve_info']),
        gr.Slider(0,2, label=texts['alpha_s'], step=0.01, value=1.0, info=texts['alpha_s_info']),
        gr.Slider(0.1,2, label=texts['alpha_i'], step=0.01, value=1.0, info=texts['alpha_i_info'])
    ],
    outputs=[
        gr.Image(label=texts['result'], type="pil"),
        gr.Textbox(label=texts['niqe'], info=texts['niqe_info']),
        gr.Textbox(label=texts['brisque'], info=texts['brisque_info'])
    ],
    title=texts['title'],
    allow_flagging="never"
)

interface.launch(server_port=7862)
