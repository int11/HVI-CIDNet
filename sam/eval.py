import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import *
from loss.losses import *
from net.CIDNet import CIDNet
from measure import metrics
import dist
from data.options import option, load_datasets
from net.CIDNet_sam import CIDNet as CIDNet_sam


def eval(model, testing_data_loader, use_GT_mean=False, alpha_predict=True):
    torch.set_grad_enabled(False)
    
    model = dist.de_parallel(model)
    temp_alpha_predict = model.alpha_predict
    model.alpha_predict = alpha_predict
    
    model.eval()

    output_list = []  # 출력 이미지 저장용 리스트
    gt_list = []   # 라벨 이미지 저장용 리스트

    for batch in testing_data_loader:
        with torch.no_grad():
            input, gt, name = batch[0], batch[1], batch[2]
            input = input.cuda()
            output = model(input)
        output = torch.clamp(output.cuda(),0,1).cuda()
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_list.append(output_np)
        # gt는 tensor이므로 PIL로 변환
        from torchvision.transforms import ToPILImage
        gt_img = ToPILImage()(gt.squeeze(0).cpu())
        gt_list.append(gt_img)
        torch.cuda.empty_cache()
    
    # metrics 계산 및 반환
    avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
    
    
    torch.set_grad_enabled(True)
    model.alpha_predict = temp_alpha_predict
    return avg_psnr, avg_ssim, avg_lpips
    
if __name__ == '__main__':
    parser = option()
    parser.add_argument('--weight_path', type=str, default='./weights/train2025-09-28-190740/epoch_100.pth', help='Path to the pre-trained model weights')

    args = parser.parse_args()

    training_data_loader, testing_data_loader = load_datasets(args)

    
    eval_net = CIDNet_sam().cuda()
    # Load model weights if provided
    checkpoint_data = torch.load(args.weight_path, map_location=lambda storage, loc: storage)
    eval_net.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Loaded checkpoint from {args.weight_path}")


    print(eval(eval_net, testing_data_loader, use_GT_mean=True, alpha_predict=False))