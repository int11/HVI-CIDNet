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

def eval(model, testing_data_loader, opt, alpha_i=1.0, use_GT_mean=False, unpaired=False):
    torch.set_grad_enabled(False)
    
    model = dist.de_parallel(model)

    print('Pre-trained model is loaded.')
    
    model.eval()
    print('Evaluation:')
    
    # opt에서 필요한 값들 가져오기
    LOL = (opt.dataset == 'lol_v1')
    v2 = (opt.dataset == 'lolv2_real')
    norm_size = not (opt.dataset in ['SICE_mix', 'SICE_grad'])
    
    # label_dir 설정
    label_dir_dict = {
        'lol_v1': opt.data_valgt_lol_v1,
        'lolv2_real': opt.data_valgt_lolv2_real,
        'lolv2_syn': opt.data_valgt_lolv2_syn,
        'lol_blur': opt.data_valgt_lol_blur,
        'SID': opt.data_valgt_SID,
        'SICE_mix': opt.data_valgt_SICE_mix,
        'SICE_grad': opt.data_valgt_SICE_grad
    }
    label_dir = label_dir_dict.get(opt.dataset, None)
    
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha_i = alpha_i
    elif unpaired:
        model.trans.gated2 = True
        model.trans.alpha_i = alpha_i
    output_list = []  # 출력 이미지 저장용 리스트
    gt_list = []   # 라벨 이미지 저장용 리스트
    
    for batch in testing_data_loader:
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            output = model(input) 
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        # Convert tensor to numpy array for metrics
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_list.append(output_np)
        
        # Load corresponding ground truth image
        gt_path = os.path.join(label_dir, name[0])
        if os.path.exists(gt_path):
            from PIL import Image
            gt_img = Image.open(gt_path).convert('RGB')
            gt_list.append(gt_img)
        
        torch.cuda.empty_cache()
    # metrics 계산 및 반환
    avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
    
    # 설정 복원
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    
    torch.set_grad_enabled(True)
    return avg_psnr, avg_ssim, avg_lpips
    
if __name__ == '__main__':
    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--perc', action='store_true', help='trained with perceptual loss')
    eval_parser.add_argument('--lol', action='store_true', help='output lolv1 dataset', default=True)
    eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
    eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')
    eval_parser.add_argument('--SICE_grad', action='store_true', help='output SICE_grad dataset')
    eval_parser.add_argument('--SICE_mix', action='store_true', help='output SICE_mix dataset')

    eval_parser.add_argument('--best_GT_mean', action='store_true', help='output lol_v2_real dataset best_GT_mean')
    eval_parser.add_argument('--best_PSNR', action='store_true', help='output lol_v2_real dataset best_PSNR')
    eval_parser.add_argument('--best_SSIM', action='store_true', help='output lol_v2_real dataset best_SSIM')

    eval_parser.add_argument('--custome', action='store_true', help='output custome dataset')
    eval_parser.add_argument('--custome_path', type=str, default='./YOLO')
    eval_parser.add_argument('--unpaired', action='store_true', help='output unpaired dataset')
    eval_parser.add_argument('--DICM', action='store_true', help='output DICM dataset')
    eval_parser.add_argument('--LIME', action='store_true', help='output LIME dataset')
    eval_parser.add_argument('--MEF', action='store_true', help='output MEF dataset')
    eval_parser.add_argument('--NPE', action='store_true', help='output NPE dataset')
    eval_parser.add_argument('--VV', action='store_true', help='output VV dataset')
    eval_parser.add_argument('--alpha', type=float, default=1.0)
    eval_parser.add_argument('--unpaired_weights', type=str, default='./weights/LOLv2_syn/w_perc.pth')

    ep = eval_parser.parse_args()

    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv1/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        if ep.perc:
            weight_path = './weights/LOLv1/w_perc.pth'
        else:
            weight_path = './weights/LOLv1/wo_perc.pth'
        
            
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        if ep.best_GT_mean:
            weight_path = './weights/LOLv2_real/w_perc.pth'
            alpha = 0.84
        elif ep.best_PSNR:
            weight_path = './weights/LOLv2_real/best_PSNR.pth'
            alpha = 0.8
        elif ep.best_SSIM:
            weight_path = './weights/LOLv2_real/best_SSIM.pth'
            alpha = 0.82
            
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        if ep.perc:
            weight_path = './weights/LOLv2_syn/w_perc.pth'
        else:
            weight_path = './weights/LOLv2_syn/wo_perc.pth'
            
    elif ep.SICE_grad:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        weight_path = './weights/SICE.pth'
        
    elif ep.SICE_mix:
        eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        weight_path = './weights/SICE.pth'
    
    elif ep.unpaired: 
        if ep.DICM:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
        elif ep.LIME:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
        elif ep.MEF:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
        elif ep.NPE:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
        elif ep.VV:
            eval_data = DataLoader(dataset=get_SICE_eval_set("./datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
        elif ep.custome:
            eval_data = DataLoader(dataset=get_SICE_eval_set(ep.custome_path), num_workers=num_workers, batch_size=1, shuffle=False)
        alpha = ep.alpha
        norm_size = False
        weight_path = ep.unpaired_weights
        
    eval_net = CIDNet().cuda()

    eval(eval_net, eval_data, weight_path, ep, alpha_i=alpha, use_GT_mean=True, unpaired=ep.unpaired)

