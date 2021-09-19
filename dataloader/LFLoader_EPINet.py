import glob, scipy, os
import torch
import scipy.io
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .util import load_LFdata
from torch.utils.data import Dataset, DataLoader




class LFDataset(Dataset):
    def __init__(self,mode,Setting02_AngualrViews, data_root = 'hci_dataset/', **kwargs):
        """ LF dataset loading LF from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]
            Setting02_AngualrViews ([type]): [description]

        Optional args:
            input_size ([type]): [if mode is train, input_size should be e.g. 25 and label size should be 25-22.]
            label_size ([type]): [if mode is train_full or test_full, input_size and label size are not needed.]
        """
        if mode== 'train' or mode == 'train_full':
            self.dir_LFimages = [
        'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
        'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
        'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
        'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]
        elif mode == 'test_full':
            self.dir_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
        else:
            raise NotImplementedError
        if mode == 'train':
            self.input_size = kwargs.pop('input_size')
            self.label_size = kwargs.pop('label_size')
        if len(kwargs) > 0:
            raise ValueError('Unused keyword arguments!')
        self.data_root = data_root
        self.mode = mode
        self.Setting02_AngualrViews = Setting02_AngualrViews
        
    def __len__(self):
        return len(self.dir_LFimages)
    def __getitem__(self,idx):
        #         input_size=23+2         # Input size should be greater than or equal to 23
        # label_size=input_size-22 # Since label_size should be greater than or equal to 1
        # Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 ) 
        dir_LFimage = self.dir_LFimages[idx]
        traindata, traindata_label = load_LFdata(dir_LFimage, self.data_root) # #512x512x9x9x3, 512x512
        if self.mode == 'train':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata_for_train(traindata,
                                            traindata_label, self.input_size, self.label_size, self.Setting02_AngualrViews)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata512(traindata, traindata_label, self.Setting02_AngualrViews)
        return data_90d,data_0d,data_45d,data_m45d, data_label


class LFDataset_PreLoad(Dataset):
    def __init__(self,mode,input_size,label_size,Setting02_AngualrViews, data_root = 'hci_dataset/', **kwargs):
        """ LF dataset pre-loading all LF from disk into RAM and then process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]
            Setting02_AngualrViews ([type]): [description]
        Optional args:
            input_size ([type]): [if mode is train, input_size should be e.g. 25 and label size should be 25-22.]
            label_size ([type]): [if mode is train_full or test_full, input_size and label size are not needed.]
        """
        if mode== 'train' or mode == 'train_full':
            self.dir_LFimages = [
        'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
        'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
        'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
        'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]
        elif mode == 'test_full':
            self.dir_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
        else:
            raise NotImplementedError
        if mode == 'train':
            self.input_size = kwargs.pop('input_size')
            self.label_size = kwargs.pop('label_size')
        assert len(kwargs) == 0 
        self.data_root = data_root
        self.mode = mode
        self.Setting02_AngualrViews = Setting02_AngualrViews
        self.PreLoading() #PreLoading all LF data to RAM

    def PreLoading(self):
        N = len(self.dir_LFimages)
        #Load one sample to get shapes
        H, W, nv, nu, C = load_LFdata(self.dir_LFimages[0], self.data_root)[0].shape

        self.traindata = np.zeros([N, H, W, nv, nu, C])
        self.traindata_label = np.zeros([N,H,W])
        for i in range(N):
            self.traindata[i], self.traindata_label[i] = load_LFdata(self.dir_LFimages[i], self.data_root)  #512x512x9x9x3, 512x512
          
    def __len__(self):
        return len(self.dir_LFimages)

    def __getitem__(self,idx):
        #         input_size=23+2         # Input size should be greater than or equal to 23
        # label_size=input_size-22 # Since label_size should be greater than or equal to 1
        # Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 ) 
        
        traindata, traindata_label = self.traindata[idx], self.traindata_label[idx]
        if self.mode == 'train':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata_for_train(traindata,
                                            traindata_label, self.input_size, self.label_size, self.Setting02_AngualrViews)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata512(traindata, traindata_label, self.Setting02_AngualrViews)
        return data_90d,data_0d,data_45d,data_m45d, data_label


class LFDataset_DDFF(Dataset):
    def __init__(self,mode,Setting02_AngualrViews, data_root = 'DDFF_dataset', list_root = 'DDFF_dataset/lists/EPINet', **kwargs):
        """ LF dataset loading DDFF LF from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]
            Setting02_AngualrViews ([type]): [description]
        Optional args:
            input_size ([type]): [if mode is train, input_size should be e.g. 25 and label size should be 25-22.]
            label_size ([type]): [if mode is train_full or test_full, input_size and label size are not needed.]
        """

        if mode== 'train' or mode == 'train_full':
            with open(os.path.join(list_root,'train.list'),"r") as f:
                self.sample_list = f.readlines()
        elif mode == 'test_full':
            with open(os.path.join(list_root,'val.list'),"r") as f:
                self.sample_list = f.readlines()     
        else:
            raise NotImplementedError
        if mode == 'train':
            self.input_size = kwargs.pop('input_size')
            self.label_size = kwargs.pop('label_size')
        assert len(kwargs) == 0 
        self.data_root = data_root
        self.mode = mode
        self.Setting02_AngualrViews = Setting02_AngualrViews
        self.__CamParamInit__(calib_mat = 'DDFF_dataset/ddff-toolbox/caldata/lfcalib/IntParamLF.mat')

    def __CamParamInit__(self, calib_mat = 'DDFF_dataset/ddff-toolbox/caldata/lfcalib/IntParamLF.mat'):
        """Loading camera calibration parameter for depth to disparity conversion.
        """
        mat = scipy.io.loadmat(calib_mat)
        if 'IntParamLF' in mat:
            mat = np.squeeze(mat['IntParamLF'])
        else:
            return
        K2 = mat[1]
        fxy = mat[2:4]
        flens = max(fxy)
        fsubaperture = 521.4052 # pixel
        baseline = K2/flens*1e-3 # meters
        self.baseL_times_fsub = baseline*fsubaperture 

    
    def depth2disp(self, depth):
        return self.baseL_times_fsub/depth

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,idx):
        # input_size=23+2         # Input size should be greater than or equal to 23
        # label_size=input_size-22 # Since label_size should be greater than or equal to 1
        # Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 ) 
        LF_path, depth_path = self.sample_list[idx].rstrip().split()
        traindata = np.load(os.path.join(self.data_root,LF_path)).transpose(2,3,0,1,4) #np uint8 383x552x9,9x3. H,W,v,u,3
        depth = np.array(Image.open(os.path.join(self.data_root, depth_path)), dtype=np.float32) * 0.001 #np.float32, 383X552. 
        traindata_label = self.depth2disp(depth) #np.float32, 383X552. #TODO: Check all the disparities are small enough, in original implementation, it is clipped in [0.0202, 0.2825]
        
        disp_min, disp_max = traindata_label[np.logical_not(np.isinf(traindata_label))].min(), traindata_label[np.logical_not(np.isinf(traindata_label))].max()
        #print(disp_min, disp_max)
        assert(disp_min > 0.01 and disp_max < 0.45)
        if self.mode == 'train':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata_for_train(traindata,
                                            traindata_label, self.input_size, self.label_size, self.Setting02_AngualrViews, RGBAug = False, ScalingAug = False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata512(traindata, traindata_label, self.Setting02_AngualrViews)
        return data_90d,data_0d,data_45d,data_m45d, data_label

class LFDataset_CVIA(Dataset):
    def __init__(self,mode,Setting02_AngualrViews, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/dataset03/lists/EPINet', **kwargs):
        """ LF dataset loading CVIA LF from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]
            Setting02_AngualrViews ([type]): [description]
        Optional args:
            input_size ([type]): [if mode is train, input_size should be e.g. 25 and label size should be 25-22.]
            label_size ([type]): [if mode is train_full or test_full, input_size and label size are not needed.]
        """

        if mode== 'train' or mode == 'train_full':
            with open(os.path.join(list_root,'train.list'),"r") as f:
                self.sample_list = f.readlines()
        elif mode == 'test_full':
            with open(os.path.join(list_root,'val.list'),"r") as f:
                self.sample_list = f.readlines()     
        else:
            raise NotImplementedError
        if mode == 'train':
            self.input_size = kwargs.pop('input_size')
            self.label_size = kwargs.pop('label_size')
        assert len(kwargs) == 0 
        self.data_root = data_root
        self.mode = mode
        self.Setting02_AngualrViews = Setting02_AngualrViews

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,idx):
        # input_size=23+2         # Input size should be greater than or equal to 23
        # label_size=input_size-22 # Since label_size should be greater than or equal to 1
        # Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # number of views ( 0~8 for 9x9 ) 
        LF_path, depth_path = self.sample_list[idx].rstrip().split()
        traindata = scipy.io.loadmat(os.path.join(self.data_root,LF_path))['lf'].transpose(2,3,0,1,4)[:,:,3:12,3:12,:]#np uint8 383x552x9x9x3. H,W,v,u,3
        traindata_label = scipy.io.loadmat(os.path.join(self.data_root, depth_path))['dm_tof'].astype(np.float32) #np.float32, 434x625. 
    
        depth_min, depth_max = traindata_label[np.logical_not(np.isnan(traindata_label))].min(), traindata_label[np.logical_not(np.isnan(traindata_label))].max()
        #print(depth_min, depth_max)
        assert depth_min > 0.2 and depth_max < 2.6, 'Get min {} and max {}'.format(depth_min, depth_max)
        if self.mode == 'train':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata_for_train(traindata,
                                            traindata_label, self.input_size, self.label_size, self.Setting02_AngualrViews, RGBAug = False, ScalingAug = False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            data_90d,data_0d,data_45d,data_m45d, data_label = generate_traindata512(traindata, traindata_label, self.Setting02_AngualrViews)
        return data_90d,data_0d,data_45d,data_m45d, data_label

def generate_traindata_for_train(traindata,traindata_label,input_size,label_size,Setting02_AngualrViews, RGBAug = True, ScalingAug= True):
    
    """
     input: traindata   (HxWx9x9x3) uint8
            traindata_label (HxW)   float32
            input_size 23~   int
            label_size 1~    int
            Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9 
            ScalingAug: Whether perform scaling augmentation
     Generate traindata using LF image and disparity map
     by randomly chosen variables.
     1.  gray image: random R,G,B --> R*img_R + G*img_G + B*imgB 
     2.  patch-wise learning: random x,y  --> LFimage[x:x+size1,y:y+size2]
     3.  scale augmentation: scale 1,2,3  --> ex> LFimage[x:x+2*size1:2,y:y+2*size2:2]
     
     output: traindata_90d   (len(Setting02_AngualrViews) x input_size x input_size ) float32        
             traindata_0d    (len(Setting02_AngualrViews) x input_size x input_size ) float32  
             traindata_45d   (len(Setting02_AngualrViews) x input_size x input_size) float32
             traindata_m45d  (len(Setting02_AngualrViews) x input_size x input_size) float32
             traindata_label_cropped (label_size x label_size)                   float32

    """
    
    H, W, _, _, _ = traindata.shape

    """ initialize image_stack & label """ 
    traindata_90d=np.zeros((len(Setting02_AngualrViews), input_size,input_size),dtype=np.float32)
    traindata_0d=np.zeros((len(Setting02_AngualrViews), input_size,input_size),dtype=np.float32)
    traindata_45d=np.zeros((len(Setting02_AngualrViews), input_size,input_size),dtype=np.float32)
    traindata_m45d=np.zeros((len(Setting02_AngualrViews), input_size,input_size),dtype=np.float32)        
    
    traindata_label_cropped=np.zeros((label_size,label_size), dtype=np.float32)
    
    """ inital variable """
    start1=Setting02_AngualrViews[0]
    end1=Setting02_AngualrViews[-1]    
    crop_half1=int(0.5*(input_size-label_size))
    
    """ Generate image stacks"""

    #continue sampling until the patch find is not constant image or an invalid region.
    """//Variable for gray conversion//"""
    if RGBAug:
        rand_3color=0.05+np.random.rand(3)
        rand_3color=rand_3color/np.sum(rand_3color) 
        R=rand_3color[0]
        G=rand_3color[1]
        B=rand_3color[2]
    else:
        R = 0.299
        G = 0.587
        B = 0.114
    """
        //Shift augmentation for 7x7, 5x5 viewpoints,.. //
        Details in our epinet paper.
    """
    if(len(Setting02_AngualrViews)==7):
        ix_rd = np.random.randint(0,3)-1
        iy_rd = np.random.randint(0,3)-1
    if(len(Setting02_AngualrViews)==9):
        ix_rd = 0
        iy_rd = 0

    if ScalingAug: 
        kk=np.random.randint(17)            
        if(kk<8):
            scale=1
        elif(kk<14):   
            scale=2
        elif(kk<17): 
            scale=3
    else:
        scale = 1

    idx_start = np.random.randint(0,H-scale*input_size)
    idy_start = np.random.randint(0,W-scale*input_size)  
    #print(idx_start, idy_start)  
       
    seq0to8=np.array(Setting02_AngualrViews)+ix_rd    
    seq8to0=np.array(Setting02_AngualrViews[::-1])+iy_rd
    
    '''
        Four image stacks are selected from LF full(HxW) images.
        gray-scaled, cropped and scaled  
        
        traindata_0d  <-- RGBtoGray( traindata[scaled_input_size, scaled_input_size, 4(center),    0to8    ] )
        traindata_90d   <-- RGBtoGray( traindata[scaled_input_size, scaled_input_size, 8to0,       4(center) ] )
        traindata_45d  <-- RGBtoGray( traindata[scaled_input_size, scaled_input_size, 8to0,         0to8    ] )
        traindata_m45d <-- RGBtoGray( traindata[scaled_input_size, scaled_input_size, 0to8,         0to8    ] )      
        '''
    traindata_0d[:,:,:]=np.squeeze(R*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),0].astype('float32')+
                                                G*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),1].astype('float32')+
                                                B*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),2].astype('float32')).transpose(2,0,1)
    
    traindata_90d[:,:,:]=np.squeeze(R*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,0].astype('float32')+
                                            G*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,1].astype('float32')+
                                            B*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,2].astype('float32')).transpose(2,0,1)
    for kkk in range(start1,end1+1):
        
        traindata_45d[kkk-start1,:,:]=np.squeeze(R*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                            G*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                            B*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                    
        traindata_m45d[kkk-start1,:,:]=np.squeeze(R*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                            G*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                            B*traindata[idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
    '''
        traindata_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
        '''                

    traindata_label_cropped[:,:]=(1.0/scale)*traindata_label[idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
                                                                        idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale]
                                
    traindata_90d=np.float32((1/255)*traindata_90d)
    traindata_0d =np.float32((1/255)*traindata_0d)
    traindata_45d=np.float32((1/255)*traindata_45d)
    traindata_m45d=np.float32((1/255)*traindata_m45d)
    
    return traindata_90d,traindata_0d,traindata_45d,traindata_m45d, traindata_label_cropped


def generate_traindata512(traindata,traindata_label,Setting02_AngualrViews):
    """   
    Generate validation or test set( = full size LF images) 
    
     input: traindata   (HxWx9x9x3) uint8
            traindata_label (HxW)   float32
            Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9            
     
    
     output: traindata_90d   (len(Setting02_AngualrViews), H x W ) float32        
             traindata_0d    (len(Setting02_AngualrViews), H x W ) float32  
             traindata_45d   (len(Setting02_AngualrViews), H x W ) float32
             traindata_m45d  (len(Setting02_AngualrViews), H x W ) float32
             traindata_label (HxW)   float32(H x W )               float32            
    """
#        else:
    input_size_H, input_size_W = traindata_label.shape
    traindata_90d=np.zeros((len(Setting02_AngualrViews),input_size_H,input_size_W),dtype=np.float32)
    traindata_0d=np.zeros((len(Setting02_AngualrViews), input_size_H,input_size_W),dtype=np.float32)
    traindata_45d=np.zeros((len(Setting02_AngualrViews), input_size_H,input_size_W),dtype=np.float32)
    traindata_m45d=np.zeros((len(Setting02_AngualrViews), input_size_H,input_size_W),dtype=np.float32)        

    
    """ inital setting """

    # crop_half1=int(0.5*(input_size-label_size))
    start1=Setting02_AngualrViews[0]
    end1=Setting02_AngualrViews[-1]
#        starttime=time.process_time() 0.375초 정도 걸림. i5 기준

        
    R = 0.299 ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
    G = 0.587
    B = 0.114


    ix_rd = 0
    iy_rd = 0
    idx_start = 0
    idy_start = 0

    seq0to8=np.array(Setting02_AngualrViews)+ix_rd
    seq8to0=np.array(Setting02_AngualrViews[::-1])+iy_rd

    traindata_0d[:,:,:]=np.squeeze(R*traindata[idx_start: idx_start+input_size_H, idy_start: idy_start+input_size_W, 4+ix_rd, seq0to8,0].astype('float32')+
                                                G*traindata[idx_start: idx_start+input_size_H, idy_start: idy_start+input_size_W, 4+ix_rd, seq0to8,1].astype('float32')+
                                                B*traindata[idx_start: idx_start+input_size_H, idy_start: idy_start+input_size_W, 4+ix_rd, seq0to8,2].astype('float32')).transpose(2,0,1)   
                                
    
    traindata_90d[:,:,:]=np.squeeze(R*traindata[ idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, seq8to0, 4+iy_rd,0].astype('float32')+
                                            G*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, seq8to0, 4+iy_rd,1].astype('float32')+
                                            B*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, seq8to0, 4+iy_rd,2].astype('float32')).transpose(2,0,1)  
    for kkk in range(start1,end1+1):
        
        traindata_45d[kkk-start1,:,:]=np.squeeze(R*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, (8)-kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                            G*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, (8)-kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                            B*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, (8)-kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                    
        traindata_m45d[kkk-start1,:,:]=np.squeeze(R*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                            G*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                            B*traindata[idx_start: idx_start+input_size_H,idy_start: idy_start+input_size_W, kkk+ix_rd, kkk+iy_rd,2].astype('float32'))

    traindata_90d=np.float32((1/255)*traindata_90d)
    traindata_0d =np.float32((1/255)*traindata_0d)
    traindata_45d=np.float32((1/255)*traindata_45d)
    traindata_m45d=np.float32((1/255)*traindata_m45d)

    #TODO Is this necessary
    traindata_90d=np.minimum(np.maximum(traindata_90d,0),1)
    traindata_0d=np.minimum(np.maximum(traindata_0d,0),1)
    traindata_45d=np.minimum(np.maximum(traindata_45d,0),1)
    traindata_m45d=np.minimum(np.maximum(traindata_m45d,0),1)

    return traindata_90d,traindata_0d,traindata_45d,traindata_m45d, traindata_label.copy()