from dataloader.FSLoader_FSNet import FSdataset, FSdataset_DDFF, FSdataset_DDFF_down_2x, FSdataset_DDFF_down_2x_except_last, FSdataset_DDFF_blur, FSdataset_CVIA, FSdataset_CVIA_down_2x, FSdataset_CVIA_down_2x_except_last, FSdataset_CVIA_blur
from tqdm import tqdm
from torch.utils.data import DataLoader
from eval_utils.utils import eval_metrics, AverageMeter, MaskedL1Loss, MaskedL1Loss_nan
import numpy as np
import torch
import os
from networks.fsnet import FSNet, FSNet_two_resolution_v1
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
            for i_batch, sample_batched in enumerate(tqdm(Loader, desc = ' Validating {} dataset, epoch {}/{}'.format(prefix, ep, opt.num_epochs))): 
                FS, data_label = sample_batched
                if isinstance(FS, list): #For case where last sensor plane has different resolution
                    FS = [item.cuda() for item in FS]
                else:
                    FS = FS.cuda()
                output = model(FS)
                mae, bp, output482, valid_mask = eval_metrics(output.cpu().detach().numpy(), data_label.numpy(), dataset = opt.dataset)

                mean_squared_errors_x100.update(100*np.average(np.square(mae[valid_mask])), len(data_label))
                bad_pixel_ratios.update(100*np.average(bp[valid_mask]), len(data_label))
                output482_all.append(output482)

            output482_all = np.concatenate(output482_all[:num_samples_to_log], axis=1)
            writer.add_scalar(prefix+'_bad_pixel_ratio', bad_pixel_ratios.avg, ep) 
            writer.add_scalar(prefix+'_mean_squared_errors_x100', mean_squared_errors_x100.avg, ep) 
            writer.add_image(prefix+'_disparity_map', output482_all,ep)
    
    writer = SummaryWriter(opt.out_path)
    if opt.dataset == 'hci':
        train_dataset = FSdataset(mode='train', input_size = opt.input_size, depth_data_root = 'hci_dataset/', FS_data_root='FS_generated_hci_dataset/nF_7/')
        train_dataset512 = FSdataset(mode='train_full', depth_data_root = 'hci_dataset/', FS_data_root='FS_generated_hci_dataset/nF_7/')
        val_dataset512 = FSdataset(mode='test_full', depth_data_root = 'hci_dataset/', FS_data_root='FS_generated_hci_dataset/nF_7/')

    elif opt.dataset == 'DDFF':
        train_dataset = FSdataset_DDFF(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='DDFF_dataset')
        train_dataset512 = FSdataset_DDFF(mode='train_full', list_root = opt.list_root, data_root='DDFF_dataset')
        val_dataset512 = FSdataset_DDFF(mode='test_full', list_root = opt.list_root, data_root='DDFF_dataset')

    elif opt.dataset == 'DDFF_down_2x':
        train_dataset = FSdataset_DDFF_down_2x(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='DDFF_dataset')
        train_dataset512 = FSdataset_DDFF_down_2x(mode='train_full', list_root = opt.list_root, data_root='DDFF_dataset')
        val_dataset512 = FSdataset_DDFF_down_2x(mode='test_full', list_root = opt.list_root, data_root='DDFF_dataset')

    elif opt.dataset == 'DDFF_down_2x_except_last':
        train_dataset = FSdataset_DDFF_down_2x_except_last(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='DDFF_dataset')
        train_dataset512 = FSdataset_DDFF_down_2x_except_last(mode='train_full', list_root = opt.list_root, data_root='DDFF_dataset')
        val_dataset512 = FSdataset_DDFF_down_2x_except_last(mode='test_full', list_root = opt.list_root, data_root='DDFF_dataset')
    
    elif opt.dataset == 'DDFF_blur':
        train_dataset = FSdataset_DDFF_blur(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='DDFF_dataset', blur_rate=opt.blur_rate)
        train_dataset512 = FSdataset_DDFF_blur(mode='train_full', list_root = opt.list_root, data_root='DDFF_dataset', blur_rate=opt.blur_rate)
        val_dataset512 = FSdataset_DDFF_blur(mode='test_full', list_root = opt.list_root, data_root='DDFF_dataset', blur_rate=opt.blur_rate)      

    elif opt.dataset == 'CVIA':
        train_dataset = FSdataset_CVIA(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        train_dataset512 = FSdataset_CVIA(mode='train_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        val_dataset512 = FSdataset_CVIA(mode='test_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03') 

    elif opt.dataset == 'CVIA_down_2x':
        train_dataset = FSdataset_CVIA_down_2x(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        train_dataset512 = FSdataset_CVIA_down_2x(mode='train_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        val_dataset512 = FSdataset_CVIA_down_2x(mode='test_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03')    

    elif opt.dataset == 'CVIA_down_2x_except_last':
        train_dataset = FSdataset_CVIA_down_2x_except_last(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        train_dataset512 = FSdataset_CVIA_down_2x_except_last(mode='train_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03')
        val_dataset512 = FSdataset_CVIA_down_2x_except_last(mode='test_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03')            
    elif opt.dataset == 'CVIA_blur':
        train_dataset = FSdataset_CVIA_blur(mode='train', input_size = opt.input_size, list_root = opt.list_root, data_root='CVIA_dataset/dataset03', blur_rate=opt.blur_rate)
        train_dataset512 = FSdataset_CVIA_blur(mode='train_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03', blur_rate=opt.blur_rate )
        val_dataset512 = FSdataset_CVIA_blur(mode='test_full', list_root = opt.list_root, data_root='CVIA_dataset/dataset03', blur_rate=opt.blur_rate) 
    worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)) # Ensure np.random working properly in trainLoader
    trainLoader = DataLoader(train_dataset,batch_size = opt.batch_size,shuffle = True, pin_memory = True, num_workers = 2, worker_init_fn = worker_init_fn)
    trainLoader512 = DataLoader(train_dataset512,batch_size = 1,pin_memory = True,shuffle = False, num_workers = 0)
    valLoader512 = DataLoader(val_dataset512,batch_size = 1,pin_memory = True,shuffle = False, num_workers = 0)

    if opt.dataset == 'DDFF_down_2x_except_last' or opt.dataset == 'CVIA_down_2x_except_last':
        model = FSNet_two_resolution_v1(nF=opt.nF, disp_mult=opt.disp_mult, offset = opt.offset).cuda()
    else:
        model = FSNet(nF=opt.nF, disp_mult=opt.disp_mult, offset = opt.offset).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, betas=(0.9, 0.999), amsgrad=True)
    if opt.dataset == 'hci':
        criterion = torch.nn.L1Loss(reduction='mean') #Return single number average across elements and batch
    elif opt.dataset == 'DDFF' or opt.dataset == 'DDFF_blur':
        criterion = MaskedL1Loss
    elif opt.dataset == 'CVIA' or opt.dataset == 'CVIA_down_2x' or opt.dataset == 'DDFF_down_2x' or opt.dataset == 'DDFF_down_2x_except_last' or opt.dataset == 'CVIA_down_2x_except_last' or opt.dataset == 'CVIA_blur':
        criterion = MaskedL1Loss_nan

    for ep in range(opt.num_epochs):
        #train code
        model.train()
        losses = AverageMeter()
        for i_batch, sample_batched in enumerate(tqdm(trainLoader, desc = ' Training epoch {}/{}'.format(ep, opt.num_epochs))):
            FS, traindata_label = sample_batched
            if isinstance(FS, list): #For case wwhere last sensor plane has different resolution
                FS = [item.cuda() for item in FS]
            else:
                FS = FS.cuda()
            output = model(FS) # B, 1, H, W
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
    parser.add_argument('--dataset', type=str, default = 'hci', help='Either hci, DDFF(_blur) or CVIA(_blur) or DDFF_down_2x(_except_last) or CVIA_down_2x(_except_last)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10000000)
    parser.add_argument('--out_path', type=str, default='logs/trial')
    parser.add_argument('--input_size', type=int, default=125, help='Input image height and width during training')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--val_frequency', type=int, default=100, help='Validation per ?? epoch.')
    parser.add_argument('--nF', type=int, default = 7, help='size of focal stack')
    parser.add_argument('--disp_mult', type=float, default = 4, help='scaling of tanh in network disparity output. 4 for hci and 0.25 for DDFF(_blur)/DDFF_down_2x(_except_last), 0.7 for CVIA(_blur)/CVIA_down_2x(_except_last)')
    parser.add_argument('--offset', type=float, default=0, help='offset to tanh of network disparity output. 0 for hci dataset and disp_multi for DDFF(_blur)/DDFF_down_2x(_except_last), 0.9 for CVIA(_blur)/CVIA_down_2x(_except_last).')
    parser.add_argument('--list_root', type=str, default=None, help='path of list root directory which stores train.list and val.list. Only required for DDFF dataset and CVIA dataset')
    parser.add_argument('--blur_rate', type=int, default=None, help='how much times to blur the image, only used if dataset is with postfix _blur') 
    opt = parser.parse_args()


    log_path = os.path.join(opt.out_path,'logs.log')
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    hdlr = logging.FileHandler(os.path.join(opt.out_path,'logs.log'))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)

    main(opt)
