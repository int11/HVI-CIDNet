import argparse
from torch.utils.data import DataLoader
from data.data import *

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    division = 4
    parser.add_argument('--batchSize', type=int, default=int(8 / division), help='training batch size')
    parser.add_argument('--cropSize', type=int, default=400, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1500, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=bool, default=False)
    parser.add_argument('--cos_restart', type=bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--gamma', type=bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=bool, default=True, help='if gradient fluctuates too much, turn-on it')
    
    # evaluation parameters
    parser.add_argument('--use_GT_mean', type=bool, default=False, help='Use the mean of GT to rectify the output of the model')

    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--dataset', type=str, default='lol_v1', choices=['lol_v1', 'lolv2_real', 'lolv2_syn', 'lol_blur', 'SID', 'SICE_mix', 'SICE_grad'], help='Choose one dataset to train on')
    return parser


def load_datasets(opt):
    print('===> Loading datasets')
    dataset = opt.dataset
    if dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lol_v1)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lol_blur)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_eval_set(opt.data_val_SID)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    elif dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a valid dataset")
    return training_data_loader, testing_data_loader