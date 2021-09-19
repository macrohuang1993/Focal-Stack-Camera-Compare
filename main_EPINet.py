from dataloader.LFLoader_EPINet import LFDataset, LFDataset_PreLoad, LFDataset_DDFF, LFDataset_CVIA
from tqdm import tqdm
from torch.utils.data import DataLoader
from eval_utils.utils import eval_metrics, AverageMeter, MaskedL1Loss, MaskedL1Loss_nan
import numpy as np
import torch
import os
from networks.epinet import EPINet
from tensorboardX import SummaryWriter
import argparse
import logging
from eval_utils.logging_utils import *
def main(opt):

    def validate(Loader, prefix = '', num_samples_to_log=8):
        model.eval()
        with torch.no_grad():
            output482_all = []
            bad_pixel_ratios = AverageMeter()
            mean_squared_errors_x100 = AverageMeter()
            for i_batch, sample_batched in enumerate(tqdm(Loader, desc = ' Validating {} dataset, epoch {}/{}'.format(prefix,ep, opt.num_epochs))): 
                data_90d,data_0d,data_45d,data_m45d, data_label = sample_batched
                output = model(data_90d.cuda(),data_0d.cuda(),data_45d.cuda(),data_m45d.cuda())
                mae, bp, output482, valid_mask = eval_metrics(output.cpu().detach().numpy(), data_label.numpy(), dataset = opt.dataset)
                
                mean_squared_errors_x100.update(100*np.average(np.square(mae[valid_mask])), len(data_label))
                bad_pixel_ratios.update(100*np.average(bp[valid_mask]), len(data_label))
                output482_all.append(output482)

            output482_all = np.concatenate(output482_all[:num_samples_to_log], axis=1) 
            writer.add_scalar(prefix+'_bad_pixel_ratio', bad_pixel_ratios.avg, ep) 
            writer.add_scalar(prefix+'_mean_squared_errors_x100', mean_squared_errors_x100.avg, ep) 
            writer.add_image(prefix+'_disparity_map', output482_all,ep)    

    Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
    writer = SummaryWriter(opt.out_path)
    if opt.dataset == 'hci':
        train_dataset=LFDataset_PreLoad(mode='train',input_size=opt.input_size,label_size=opt.label_size,Setting02_AngualrViews= Setting02_AngualrViews, data_root = 'hci_dataset/')
        train_dataset512  = LFDataset_PreLoad(mode='train_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'hci_dataset/')
        val_dataset512 =  LFDataset_PreLoad(mode='test_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'hci_dataset/')
    elif opt.dataset == 'DDFF':
        train_dataset=LFDataset_DDFF(mode='train',input_size=opt.input_size,label_size=opt.label_size,Setting02_AngualrViews= Setting02_AngualrViews, data_root = 'DDFF_dataset')
        train_dataset512  = LFDataset_DDFF(mode='train_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'DDFF_dataset')
        val_dataset512 =  LFDataset_DDFF(mode='test_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'DDFF_dataset')
    elif opt.dataset == 'CVIA':
        train_dataset=LFDataset_CVIA(mode='train',input_size=opt.input_size,label_size=opt.label_size,Setting02_AngualrViews= Setting02_AngualrViews, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/dataset03/lists/EPINet')
        train_dataset512  = LFDataset_CVIA(mode='train_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/dataset03/lists/EPINet')
        val_dataset512 =  LFDataset_CVIA(mode='test_full',Setting02_AngualrViews= Setting02_AngualrViews,data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/dataset03/lists/EPINet')       
    worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)) # Ensure np.random working properly in trainLoader
    trainLoader = DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True, pin_memory = True, num_workers = 2, worker_init_fn = worker_init_fn)
    trainLoader512 = DataLoader(train_dataset512,batch_size = 1,pin_memory = True,shuffle = False, num_workers = 0)
    valLoader512 = DataLoader(val_dataset512,batch_size = 1,pin_memory = True,shuffle = False, num_workers = 0)

    model = EPINet(filt_num=70, view_n = len(Setting02_AngualrViews),conv_depth = 7).cuda()
    #model.load_state_dict(torch.load('/home/zyhuang/WD/LF_FS_compare/LF_depth/epinet_pytorch/logs/EPINet/CVIA_data/Epoch_12100_model.pth'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, betas=(0.9, 0.999), amsgrad=True)
    if opt.dataset == 'hci':
        criterion = torch.nn.L1Loss(reduction='mean') #Return single number average across elements and batch
    elif opt.dataset == 'DDFF':
        criterion = MaskedL1Loss
    elif opt.dataset == 'CVIA':
        criterion = MaskedL1Loss_nan
    for ep in range(opt.num_epochs):
        #train code
        model.train()
        losses = AverageMeter()
        for i_batch, sample_batched in enumerate(tqdm(trainLoader, desc = ' Training epoch {}/{}'.format(ep, opt.num_epochs))):
            traindata_90d,traindata_0d,traindata_45d,traindata_m45d, traindata_label = sample_batched
            output = model(traindata_90d.cuda(),traindata_0d.cuda(),traindata_45d.cuda(),traindata_m45d.cuda()) # B, 1, H, W
            loss = criterion(torch.squeeze(output, dim=1), traindata_label.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(loss.item())
            losses.update(loss.item(), output.shape[0])

        writer.add_scalar('train_loss', losses.avg, ep)   
        logger.info("Epoch {}, avg_loss:{}".format(ep,losses.avg))
        if ep % opt.val_frequency == 0:
            #validate code
            validate(trainLoader512, prefix = 'train')
            validate(valLoader512, prefix = 'val')
            torch.save(model.state_dict(), os.path.join(opt.out_path,'Epoch_{}_model.pth'.format(ep)))



def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    #if state['epoch'] % 10 == 0:
    torch.save(state, os.path.join(opt.outf,filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'hci', help='Either hci, DDFF or CVIA')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10000000)
    parser.add_argument('--out_path', type=str, default='logs/trial')
    parser.add_argument('--input_size', type=int, default=125, help='Input image height and width during training')
    parser.add_argument('--label_size', type=int, default=103,  help='Size of network output during training.  Same as the size of label for calculating the loss')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--val_frequency', type=int, default=100, help='Validation per ?? epoch.')
    opt = parser.parse_args()

    log_path = os.path.join(opt.out_path,'logs.log')
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    hdlr = logging.FileHandler(os.path.join(opt.out_path,'logs.log'))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)

    main(opt)




