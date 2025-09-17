import torch
import torch.nn as nn
from net.HVI_transform_sam import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from huggingface_hub import hf_hub_download
import safetensors.torch as sf


def load_sam_model(model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth", device="cuda"):
    """Hugging Face에서 SAM 모델을 다운로드하고 로드"""
    # sam_model_path를 repo_id와 filename으로 분리
    parts = model_path.split('/')
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
    return sam


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 cidnet_model_path="Fediory/HVI-CIDNet-LOLv1-woperc",
                 sam_model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
        ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        
        # Alpha prediction layer for alpha_s and alpha_i
        self.alpha_predictor = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False),  # input channels = 3 (HVI)
            nn.GroupNorm(1, ch1),
            nn.SiLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False),  # output 2 channels for alpha_s and alpha_i
        )


        if cidnet_model_path != None:
            print(f"Loading CIDNet model from: {cidnet_model_path}")

            # Hugging Face Hub에서 CIDNet model 다운로드
            model_file = hf_hub_download(
                repo_id=cidnet_model_path,
                filename="model.safetensors",
                repo_type="model"
            )
            print(f"CIDNet model downloaded from: {model_file}")
            
            # 모델 초기화 및 가중치 로드
            state_dict = sf.load_file(model_file)
            self.load_state_dict(state_dict, strict=False)

        if sam_model_path != None:
            self.sam = load_sam_model(sam_model_path)

        # Freeze all loaded parameters (before adding new layers)
        for param in self.parameters():
            param.requires_grad = False
        
        # Add new trainable layers after freezing


    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.RGB_to_HVI(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi

        # Predict alpha_s and alpha_i values
        alpha_maps = self.alpha_predictor(output_hvi)
        # Sigmoid를 통과시켜 0~1 범위로 만든 후 원하는 범위로 스케일링
        alpha_s = torch.sigmoid(alpha_maps[:, 0:1, :, :]) * 0.6 + 1.0  # 1.0~1.6
        alpha_i = torch.sigmoid(alpha_maps[:, 1:2, :, :]) * 0.4 + 0.8  # 0.8~1.2

        sam_output = self.sam(x)

        output_rgb = self.trans.HVI_to_RGB(output_hvi, alpha_s, alpha_i)

        return output_rgb
    
    def RGB_to_HVI(self,x):
        hvi = self.trans.RGB_to_HVI(x)
        return hvi
    

if __name__ == "__main__":
    model = CIDNet()
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)  # Example input tensor
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (1, 3, 256, 256) for RGB output
    print("Model loaded and tested successfully.")