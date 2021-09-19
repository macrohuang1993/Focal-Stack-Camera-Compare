import glob, os
from skimage import io
import scipy.io
import numpy as np
from .util import load_disp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage.measure import block_reduce
def load_FS(FS_list):

    FS = []
    for FS_path in FS_list:
        temp = io.imread(FS_path)[np.newaxis] # 1, H， W， C, uint8 array
        _, h,w,c = temp.shape
        assert c == 4 or c == 3
        temp = temp[:,:,:,:3]
        FS.append(temp)  
    return np.concatenate(FS,axis=0) # nF, H, W, C 

class FSdataset(Dataset):
    def __init__(self, mode, depth_data_root = 'hci_dataset/', FS_data_root='FS_generated_hci_dataset/', **kwargs):
        """ FS dataset loading FS from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]

        Optional arguments: 
            input_size ([type]): [If mode is train, input_size is required and  it is the size after crop for FS and disp.]
                                 [If mode is train_full or test_full, input_size is not required ]
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
        
        self.depth_data_root = depth_data_root
        self.FS_data_root = FS_data_root
        self.mode = mode
        if mode == 'train':
            self.input_size = kwargs.pop('input_size')
        if len(kwargs) > 0:
            raise ValueError('Unused keyword arguments!')
    

    def __len__(self):
        return len(self.dir_LFimages)
    def __getitem__(self,idx):

        dir_LFimage = self.dir_LFimages[idx]
        FS_list = sorted(glob.glob(self.FS_data_root+dir_LFimage+'/*.png'))
        FS = load_FS(FS_list) # nF, H, W, 3, uint8 np array

        data_label = load_disp(dir_LFimage, self.depth_data_root) #512x512, np float32

        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW
        
        #TODO: Try augmentation from pytorch modules?
        # sample = {'FS':FS,'LF':LF}
        # if self.transform is not None:
        #     # possible normalization and augmentation done here.
        #     sample = self.transform(sample)
        # return sample

class FSdataset_DDFF(Dataset):
    def __init__(self, mode, data_root = 'DDFF_dataset', list_root = 'DDFF_dataset/lists/FS_nF_7', **kwargs):
        """ FS dataset loading FS from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]

        Optional arguments: 
            input_size ([type]): [If mode is train, input_size is required and  it is the size after crop for FS and disp.]
                                 [If mode is train_full or test_full, input_size is not required ]
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
        if len(kwargs) > 0:
            raise ValueError('Unused keyword arguments!')

        self.mode = mode
        self.data_root=data_root
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
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        depth = np.array(Image.open(os.path.join(self.data_root, depth_path)), dtype=np.float32) * 0.001 #np.float32, 383X552. 
        data_label = self.depth2disp(depth) #np.float32, 383X552. #TODO: Check all the disparities are small enough, in original implementation, it is clipped in [0.0202, 0.2825]
        
        disp_min, disp_max = data_label[np.logical_not(np.isinf(data_label))].min(), data_label[np.logical_not(np.isinf(data_label))].max()
        #print(disp_min, disp_max)
        assert(disp_min > 0.01 and disp_max < 0.45)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW

class FSdataset_DDFF_down_2x(FSdataset_DDFF):
    def __init__(self,mode, data_root = 'DDFF_dataset', list_root = 'DDFF_dataset/lists/FS_nF_7', **kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.downscale = 2
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        depth = np.array(Image.open(os.path.join(self.data_root, depth_path)), dtype=np.float32) * 0.001 #np.float32, 383X552. 
        data_label = self.depth2disp(depth) #np.float32, 383X552. #TODO: Check all the disparities are small enough, in original implementation, it is clipped in [0.0202, 0.2825]
        
        disp_min, disp_max = data_label[np.logical_not(np.isinf(data_label))].min(), data_label[np.logical_not(np.isinf(data_label))].max()
        #print(disp_min, disp_max)
        assert(disp_min > 0.01 and disp_max < 0.45)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array

        #Downsampling depth and FS
        H,W = data_label.shape
        mask = np.logical_not(np.isinf(data_label)).astype(np.float32)
        mask = block_reduce(mask,(self.downscale,self.downscale), func=np.mean)
        data_label[np.isinf(data_label)] = 0 # Setting invalid pixel to 0 so average pooling next won't generate all nans. 
        data_label = block_reduce(data_label,(self.downscale,self.downscale), func=np.mean)
        data_label = (data_label / mask)   ## div by 0 generates nan, indicating invalid regions.
        data_label = data_label[:H//self.downscale, :W//self.downscale]
        FS = block_reduce(FS,(1,self.downscale,self.downscale, 1), func=np.mean)[:,:H//self.downscale, :W//self.downscale,:].astype(np.uint8) 

        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW

class FSdataset_DDFF_down_2x_except_last(FSdataset_DDFF):
    def __init__(self,mode, data_root = 'DDFF_dataset', list_root = 'DDFF_dataset/lists/FS_nF_7', **kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.downscale = 2
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        depth = np.array(Image.open(os.path.join(self.data_root, depth_path)), dtype=np.float32) * 0.001 #np.float32, 383X552. 
        data_label = self.depth2disp(depth) #np.float32, 383X552. #TODO: Check all the disparities are small enough, in original implementation, it is clipped in [0.0202, 0.2825]
        
        disp_min, disp_max = data_label[np.logical_not(np.isinf(data_label))].min(), data_label[np.logical_not(np.isinf(data_label))].max()
        #print(disp_min, disp_max)
        assert(disp_min > 0.01 and disp_max < 0.45)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array

        #Downsampling depth and FS
        H,W = data_label.shape
        mask = np.logical_not(np.isinf(data_label)).astype(np.float32)
        mask = block_reduce(mask,(self.downscale,self.downscale), func=np.mean)
        data_label[np.isinf(data_label)] = 0 # Setting invalid pixel to 0 so average pooling next won't generate all nans. 
        data_label = block_reduce(data_label,(self.downscale,self.downscale), func=np.mean)
        data_label = (data_label / mask)   ## div by 0 generates nan, indicating invalid regions.
        data_label = data_label[:H//self.downscale, :W//self.downscale]
        FS_back_plane = FS[0][np.newaxis][:,:(H//self.downscale) * self.downscale, :(W//self.downscale) * self.downscale,:] # Back sensor plane, should focusing on near objects
        FS_remaining = block_reduce(FS[1:],(1,self.downscale,self.downscale, 1), func=np.mean)[:,:H//self.downscale, :W//self.downscale,:].astype(np.uint8)
        if self.mode == 'train':
            FS_back_plane, FS_remaining, data_label = generate_traindata_for_train_two_resolution(FS_back_plane,FS_remaining, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS_back_plane, FS_remaining, data_label = generate_traindata512_two_resolution(FS_back_plane,FS_remaining,data_label)
        return [FS_back_plane.transpose(0,3,1,2), FS_remaining.transpose(0,3,1,2)], data_label # (1xCx2*Hx2*W,nF-1xCxHxW) and HxW

class FSdataset_DDFF_blur(FSdataset_DDFF):
    def __init__(self,mode, data_root = 'DDFF_dataset', list_root = 'DDFF_dataset/lists/FS_nF_7', blur_rate=2,**kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.blur_rate = blur_rate
        
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        FS = blur_FS(FS, self.blur_rate)
        depth = np.array(Image.open(os.path.join(self.data_root, depth_path)), dtype=np.float32) * 0.001 #np.float32, 383X552. 
        data_label = self.depth2disp(depth) #np.float32, 383X552. #TODO: Check all the disparities are small enough, in original implementation, it is clipped in [0.0202, 0.2825]
        
        disp_min, disp_max = data_label[np.logical_not(np.isinf(data_label))].min(), data_label[np.logical_not(np.isinf(data_label))].max()
        #print(disp_min, disp_max)
        assert(disp_min > 0.01 and disp_max < 0.45)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW
class FSdataset_CVIA(Dataset):
    def __init__(self, mode, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/lists/FS_nF_7', **kwargs):
        """ FS dataset loading FS from disk and process it on the fly. 
        Args:
            mode ([type]): [Either train, train_full, test_full]

        Optional arguments: 
            input_size ([type]): [If mode is train, input_size is required and  it is the size after crop for FS and disp.]
                                 [If mode is train_full or test_full, input_size is not required ]
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
        if len(kwargs) > 0:
            raise ValueError('Unused keyword arguments!')

        self.mode = mode
        self.data_root=data_root

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        data_label = scipy.io.loadmat(os.path.join(self.data_root, depth_path))['dm_tof'].astype(np.float32) #np.float32, 434x625. 

        depth_min, depth_max = data_label[np.logical_not(np.isnan(data_label))].min(), data_label[np.logical_not(np.isnan(data_label))].max()
        #print(disp_min, disp_max)
        assert depth_min > 0.2 and depth_max < 2.6, 'Get min {} and max {}'.format(depth_min, depth_max)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW

class FSdataset_CVIA_down_2x(FSdataset_CVIA):
    def __init__(self, mode, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/lists/FS_nF_7', **kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.downscale = 2
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        data_label = scipy.io.loadmat(os.path.join(self.data_root, depth_path))['dm_tof'].astype(np.float32) #np.float32, 434x625. 

        depth_min, depth_max = data_label[np.logical_not(np.isnan(data_label))].min(), data_label[np.logical_not(np.isnan(data_label))].max()
        #print(disp_min, disp_max)
        assert depth_min > 0.2 and depth_max < 2.6, 'Get min {} and max {}'.format(depth_min, depth_max)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array    

        #Downsampling depth and FS
        H,W = data_label.shape
        mask = np.logical_not(np.isnan(data_label)).astype(np.float32)
        mask = block_reduce(mask,(self.downscale,self.downscale), func=np.mean)
        data_label[np.isnan(data_label)] = 0 # Setting invalid pixel to 0 so average pooling next won't generate all nans. 
        data_label = block_reduce(data_label,(self.downscale,self.downscale), func=np.mean)
        data_label = (data_label / mask)   ## div by 0 generates nan, indicating invalid regions.
        data_label = data_label[:H//self.downscale, :W//self.downscale]
        FS = block_reduce(FS,(1,self.downscale,self.downscale, 1), func=np.mean)[:,:H//self.downscale, :W//self.downscale,:].astype(np.uint8) 

        
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW

class FSdataset_CVIA_down_2x_except_last(FSdataset_CVIA):
    def __init__(self, mode, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/lists/FS_nF_7', **kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.downscale = 2
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        data_label = scipy.io.loadmat(os.path.join(self.data_root, depth_path))['dm_tof'].astype(np.float32) #np.float32, 434x625. 

        depth_min, depth_max = data_label[np.logical_not(np.isnan(data_label))].min(), data_label[np.logical_not(np.isnan(data_label))].max()
        #print(disp_min, disp_max)
        assert depth_min > 0.2 and depth_max < 2.6, 'Get min {} and max {}'.format(depth_min, depth_max)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array    

        #Downsampling depth and FS
        H,W = data_label.shape
        mask = np.logical_not(np.isnan(data_label)).astype(np.float32)
        mask = block_reduce(mask,(self.downscale,self.downscale), func=np.mean)
        data_label[np.isnan(data_label)] = 0 # Setting invalid pixel to 0 so average pooling next won't generate all nans. 
        data_label = block_reduce(data_label,(self.downscale,self.downscale), func=np.mean)
        data_label = (data_label / mask)   ## div by 0 generates nan, indicating invalid regions.
        data_label = data_label[:H//self.downscale, :W//self.downscale]
        FS_back_plane = FS[0][np.newaxis][:,:(H//self.downscale) * self.downscale, :(W//self.downscale) * self.downscale,:] # Back sensor plane, should focusing on near objects
        FS_remaining = block_reduce(FS[1:],(1,self.downscale,self.downscale, 1), func=np.mean)[:,:H//self.downscale, :W//self.downscale,:].astype(np.uint8)

        if self.mode == 'train':
            FS_back_plane, FS_remaining, data_label = generate_traindata_for_train_two_resolution(FS_back_plane,FS_remaining, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS_back_plane, FS_remaining, data_label = generate_traindata512_two_resolution(FS_back_plane,FS_remaining,data_label)
        return [FS_back_plane.transpose(0,3,1,2), FS_remaining.transpose(0,3,1,2)], data_label # (1xCx2*Hx2*W,nF-1xCxHxW) and HxW

class FSdataset_CVIA_blur(FSdataset_CVIA):
    def __init__(self, mode, data_root = 'CVIA_dataset/dataset03', list_root = 'CVIA_dataset/lists/FS_nF_7',blur_rate=2, **kwargs):
        super().__init__(mode, data_root = data_root, list_root = list_root, **kwargs)
        self.blur_rate=blur_rate
        
    def __getitem__(self,idx):
        FS_dir, depth_path = self.sample_list[idx].rstrip().split()
        FS_file_list = sorted(glob.glob(os.path.join(self.data_root,FS_dir,'*.png')))
        FS = load_FS(FS_file_list) # nF, H, W, 3, uint8 np array
        FS = blur_FS(FS,self.blur_rate)
        data_label = scipy.io.loadmat(os.path.join(self.data_root, depth_path))['dm_tof'].astype(np.float32) #np.float32, 434x625. 

        depth_min, depth_max = data_label[np.logical_not(np.isnan(data_label))].min(), data_label[np.logical_not(np.isnan(data_label))].max()
        #print(disp_min, disp_max)
        assert depth_min > 0.2 and depth_max < 2.6, 'Get min {} and max {}'.format(depth_min, depth_max)
        #FS: nF x input_size x input_size x 3, float32 np array
        #data_label: H,w float32 np array
        if self.mode == 'train':
            FS, data_label = generate_traindata_for_train(FS, data_label, self.input_size, ScalingAug=False)
        elif self.mode == 'train_full' or self.mode == 'test_full':
            FS, data_label = generate_traindata512(FS,data_label)
        return FS.transpose(0,3,1,2), data_label # nFxCxHxW and HxW
def generate_traindata_for_train(FS, data_label, input_size, ScalingAug = True):
    """
    Args:
        FS ([type]): [nF, H, W, 3, uint8 np array]
        data_label ([type]): [HxW, np float32]
        input_size ([type]): [size after crop]
        ScalingAug: Whether perform scaling augmentation
    Returns:
        [FS_out]: [[nFxinput_sizexinput_sizex3, float32 np array]]
        [data_label_out]: [input_sizexinput_size, np float32]
    """
    nF,H,W,C = FS.shape

    """//Variable for gray conversion//"""
    #TODO: RGB flicker agumentation

    #Scaling augmentation
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

    idw_start = np.random.randint(0,W-scale*input_size)
    idh_start = np.random.randint(0,H-scale*input_size)  
    
    FS_out = FS[:, idh_start: idh_start+scale*input_size:scale,  idw_start: idw_start+scale*input_size:scale,:].astype(np.float32)
    FS_out = FS_out / 255.

    data_label_out = (1.0/scale) * data_label[idh_start: idh_start+scale*input_size:scale,  idw_start: idw_start+scale*input_size:scale]
    return FS_out, data_label_out.copy()

def generate_traindata_for_train_two_resolution(FS_backplane, FS_remaining, data_label, input_size, ScalingAug = True):
    """ version of generate_traindata_for_train when input has different resolution across focal stack
    Args:
        FS_backplane ([type]): [1, downscale*H, downscale*W, 3, uint8 np array]
        FS_remaining ([type]): [nF-1, H, W, 3, uint8 np array]

        data_label ([type]): [HxW, np float32]
        input_size ([type]): [size after crop]
        ScalingAug: Whether perform scaling augmentation
    Returns:
        [FS_out_backplane]: [[1 x downscale*input_size x downscale*input_sizex3, float32 np array]]
        [FS_out_remaining]: [[nF-1xinput_sizexinput_sizex3, float32 np array]]
        [data_label_out]: [input_sizexinput_size, np float32]
    """
    downscale = 2
    _,H,W,C = FS_remaining.shape

    """//Variable for gray conversion//"""
    #TODO: RGB flicker agumentation

    #Scaling augmentation
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

    idw_start = np.random.randint(0,W-scale*input_size)
    idh_start = np.random.randint(0,H-scale*input_size)  
    
    FS_out_remaining = FS_remaining[:, idh_start: idh_start+scale*input_size:scale,  idw_start: idw_start+scale*input_size:scale,:].astype(np.float32)
    FS_out_remaining = FS_out_remaining / 255.

    FS_out_backplane = FS_backplane[:, downscale*idh_start: downscale*idh_start+downscale*scale*input_size:scale,  downscale*idw_start: downscale*idw_start+downscale*scale*input_size:scale,:].astype(np.float32)
    FS_out_backplane = FS_out_backplane / 255.

    data_label_out = (1.0/scale) * data_label[idh_start: idh_start+scale*input_size:scale,  idw_start: idw_start+scale*input_size:scale]
    return FS_out_backplane, FS_out_remaining, data_label_out.copy()

def generate_traindata512(FS, data_label):
    """
    Args:
        FS ([type]): [nF, H, W, 3, uint8 np array]
        data_label ([type]): [HxW, np float32]

    Returns:
        [FS_out]: [[nFxHxWx3, float32 np array]]
        [data_label_out]: [HxW, np float32]
    """

    FS_out = FS.astype(np.float32)/ 255.
    data_label_out = data_label.copy()
    return FS_out, data_label_out

def generate_traindata512_two_resolution(FS_backplane, FS_remaining, data_label):
    """
    Args:
        FS_backplane ([type]): [1, downscale*H, downscale*W, 3, uint8 np array]
        FS_remaining ([type]): [nF-1, H, W, 3, uint8 np array]
        data_label ([type]): [HxW, np float32]

    Returns:
        [FS_out_backplane]: [[1 x downscale*H x downscale*Wx3, float32 np array]]
        [FS_out_remaining]: [[nF-1xHxWx3, float32 np array]]
        [data_label_out]: [HxW, np float32]
    """

    FS_out_backplane = FS_backplane.astype(np.float32)/ 255.
    FS_out_remaining = FS_remaining.astype(np.float32)/ 255.
    data_label_out = data_label.copy()
    return FS_out_backplane, FS_out_remaining, data_label_out

def blur_FS(FS,blur_rate):
    nF,H,W,C = FS.shape
    FS=np.pad(FS,((0,0),(0,blur_rate-H%blur_rate),(0,blur_rate-W%blur_rate),(0,0)), mode='constant')
    for i in range(blur_rate):
        for j in range(blur_rate):
            if i == 0 and j == 0:
                continue
            FS[:,i::blur_rate,j::blur_rate,:] = FS[:,::blur_rate, ::blur_rate,:]
    # FS[:,1::2,1::2,:]=FS[:,::2,::2,:]
    # FS[:,::2,1::2,:]=FS[:,::2,::2,:]
    # FS[:,1::2,::2,:]=FS[:,::2,::2,:]
    return FS[:,:H,:W,:]