#For Generating Focal stack dataset using add shift algorithm written by myself. (based on code in DDFF/FSNet)
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import os
from dataloader.util import load_LFdata, read_parameters
import configparser
# lf_HW = [372, 540] #dimensions of Lytro light fields, H,W, the 7 by 7 SAI is used out of 14 by 14 (save as ICASSP paper, see data_utils.read_lytroLF_as5D)
#Note original Lytro LF has dimension 376X541 X 14 X 14


def bilinear_interpolate(im, x, y):
    """
    Input: im (H,W)
    x,y: index point or 2D grid (could be float number), in range 0,1,...(H-1) or (W-1) to interpolate
    
    Output: interpolated image at point or input 2D grid
    
    eg: 
    im = np.random.rand(100,200)
    x = [1,5]
    y= [2,55,22]
    X, Y= np.meshgrid(x,y,indexing = 'xy')
    bilinear_interpolate(im, X, Y)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)
    
    
    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def generate_FS(LF,disparities,FSview_rounding = True):
    """
    Input：
    LF： LF array in shape of 3,H,W,nv,nu. Increasing idx of v should move camera down, increasing idx of u should move camera right.
    disparities: list of disparity at which to focus, the length gives the size of FS, i.e. nF. 
    Disparity <0 means refocusing farther than the focal plane
    FSview_rounding: Whether generate FS at integer u,v view point floor rounded (True) or just the true Center view (False).
    Output:
    refocused images in shape of nF,C,H,W
    """
    
    nC,H,W,nv,nu = LF.shape
    nF = len(disparities)
    FS = np.zeros([nF,3,H,W])
    I_tmp = np.zeros([3,H,W])
    
    X, Y = np.meshgrid(range(W),range(H),indexing = 'xy')
    for iF in range(nF):
        disp = disparities[iF]
        #print(iF)
        for iv in range(nv):
            for iu in range(nu):
                for ic in range(nC):
                    #I_tmp[ic] = bilinear_interpolate(LF[ic,:,:,iv,iu], X-disp*(iu-(nu-1)/2), Y+disp*(iv-(nv-1)/2)) # for MIT LF in https://github.com/MITComputationalCamera/LightFields, there is a sign difference because the LF given has opposite covention of positive v direction
                    if FSview_rounding:
                        I_tmp[ic] = bilinear_interpolate(LF[ic,:,:,iv,iu], X-disp*(iu-np.floor((nu-1)/2)), Y-disp*(iv-np.floor((nv-1)/2))) #For Flower CVPR LF dataset
                    else:
                        I_tmp[ic] = bilinear_interpolate(LF[ic,:,:,iv,iu], X-disp*(iu-(nu-1)/2), Y-disp*(iv-(nv-1)/2)) #For Flower CVPR LF dataset                       
                FS[iF] += I_tmp/(nv*nu)
    return np.clip(FS,0,255).astype(np.uint8)


def generate_FSdataset(dir_LFimages, dmin, dmax, nF=7, data_root='hci_dataset/', save_root = 'FS_generated_hci_dataset'):
    for sample_dir in dir_LFimages:
        # Return [H,W,nv,nu,3], [H,W], Increasing idx of v should move camera down, increasing idx of u should move camera right. 
        # which holds for flower light field dataset and hci light field benchmark dataset. 
        LFdata, label = load_LFdata(sample_dir, data_root = data_root)
        params=read_parameters(os.path.join(data_root,sample_dir))
        disp_min, disp_max = params['disp_min'], params['disp_max']
        print(sample_dir, disp_min,disp_max)
        disps=np.linspace(dmin,dmax,nF)
        FS = generate_FS(LFdata.transpose(4,0,1,2,3), disps, FSview_rounding = False)
        for idx, image in enumerate(FS):
            temp_dir = save_root+ '/' + sample_dir
            os.makedirs(temp_dir, exist_ok=True)
            plt.imsave(temp_dir + '/{}.png'.format(idx),  image.transpose(1,2,0))
        #TODO test the config writting part below
        config = configparser.ConfigParser()
        config['FS_Setting'] = {'FS_disp_min':dmin, 'FS_disp_max':dmax}
        with open(temp_dir+'/setting.cfg', 'w') as configfile:
            config.write(configfile)


#TODO Write an adaptive disp_min dispmax for focal stack generation and use a disp aware network structure, i.e., feed additional disp channels. 
if __name__ == "__main__":

    train_dir_LFimages = [
            'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
            'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
            'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
            'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]
    test_dir_LFimages = [
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']

    dmin=-3
    dmax=3
    nF = 14
    print('Generating FS for train.')
    generate_FSdataset(train_dir_LFimages,dmin,dmax, nF, save_root = 'FS_generated_hci_dataset/nF_14')
    print('Generating FS for test.')
    generate_FSdataset(test_dir_LFimages,dmin,dmax, nF, save_root = 'FS_generated_hci_dataset/nF_14')
    print('Done')