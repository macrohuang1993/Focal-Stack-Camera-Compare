import numpy as np
import imageio, torch
import torch.nn.functional as F
def MaskedL1Loss(input_disp, target_disp):
    """ Calculating loss only on region with valid gt_disparity (not inf)
    Args:
        input_disp ([type]): [description]
        target_disp ([type]): [description]
    """
    target_valid = (1-torch.isinf(target_disp))
    return F.l1_loss(input_disp[target_valid], target_disp[target_valid], reduction = 'mean')
def MaskedL1Loss_nan(input_disp, target_disp):
    """ Calculating loss only on region with valid gt (not nan)
    Args:
        input_disp ([type]): [description]
        target_disp ([type]): [description]
    """
    target_valid = (1-torch.isnan(target_disp))
    return F.l1_loss(input_disp[target_valid], target_disp[target_valid], reduction = 'mean')

def eval_metrics(train_output, traindata_label, dataset = ''):
    """Evaluate the MAE error and bad pixel rate for the network output, on the central (H-30)x(W-30) region.

    Args:
        train_output ([numpy array]): [B,1,H-22,W-22] for EPINet or [B,1,H,W] for FSNet.
        traindata_label ([numpy array]): [B,H,W]
        dataset ([string]): Either HCI, DDFF or CVIA
    Returns:
        mae, bp[float]: [MAE error and bad pixel rate] of shape B,H-30, W-30
        train_output482_all[numpy array]:[(H-30)*2, 16 * (W-30)] result disp for visualization
        valid_mask: boolean mask indicating whether the pixel has valid gt.
    """
    if dataset == 'HCI':
        sparse = False
        scalef, offset = 25, 100
    elif dataset == 'DDFF' or dataset == 'DDFF_down_2x' or dataset == 'DDFF_down_2x_except_last' or dataset == 'DDFF_blur':
        sparse = True
        scalef, offset = 500, 0
    elif dataset == 'CVIA' or dataset == 'CVIA_down_2x' or dataset == 'CVIA_down_2x_except_last' or dataset == 'CVIA_blur':
        sparse = True
        scalef, offset = 160, 0 
    sz, H, W=traindata_label.shape
    train_output=np.squeeze(train_output,axis=1)

    pad1_half=int(0.5*(np.size(traindata_label,1)-np.size(train_output,1)))
    train_output_full = np.pad(train_output, ((0,0),(pad1_half,pad1_half),(pad1_half,pad1_half)), 'constant',)
    
    train_label_central=traindata_label[:,15:-15,15:-15]
    train_output_central = train_output_full[:,15:-15, 15:-15]

    mae=np.abs(train_output_central-train_label_central)
    train_bp=(mae>=0.07)

    train_outputCentral_all=np.zeros((2*(H-30),sz*(W-30)),np.uint8)        
    train_outputCentral_all[0:(H-30),:]=np.uint8(scalef*np.reshape(np.transpose(train_label_central,(1,0,2)),(H-30,sz*(W-30)))+offset)
    train_outputCentral_all[(H-30):2*(H-30),:]=np.uint8(scalef*np.reshape(np.transpose(train_output_central,(1,0,2)),(H-30,sz*(W-30)))+offset)
    
    if sparse:
        valid_mask = np.logical_not(np.logical_or(np.isinf(train_label_central), np.isnan(train_label_central))) #Pixel not inf or not nan
    else:
        valid_mask = np.ones((sz, H-30, W-30), dtype=bool)
    return mae, train_bp, train_outputCentral_all, valid_mask 
    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    a=np.random.rand(3,1,490,490)
    b=np.random.rand(3,512,512)
    r1 = eval_metrics(a,b)
    #r2=0
    r2 = eval_metrics(a,b)
    print([np.array_equal(r1[i],r2[i]) for i in range(3)])