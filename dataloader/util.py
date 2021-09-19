import numpy as np
import imageio
import os

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    
    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line
    
    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data
        
def load_disp(dir_LFimage, data_root = 'hci_dataset/'):
    traindata_label=np.zeros((512, 512),np.float32)
    try:            
        traindata_label  = np.float32(read_pfm(data_root+dir_LFimage+'/gt_disp_lowres.pfm')) # load LF disparity map
    except:
        print(data_root+dir_LFimage+'/gt_disp_lowres.pfm..does not exist' % i )            
        
    return traindata_label  

def load_LFdata(dir_LFimage, data_root = 'hci_dataset/'):    
    traindata=np.zeros((512, 512, 9, 9, 3),np.uint8)
    traindata_label=np.zeros((512, 512),np.float32)

    #print(dir_LFimage)
    for i in range(81):
        try:
            tmp  = np.float32(imageio.imread(data_root+dir_LFimage+'/input_Cam0%.2d.png' % i)) # load LF images(9x9) 
        except:
            print(data_root+dir_LFimage+'/input_Cam0%.2d.png..does not exist' % i )
        traindata[:,:,i//9,i-9*(i//9),:]=tmp  
        del tmp
    try:            
        traindata_label  = np.float32(read_pfm(data_root+dir_LFimage+'/gt_disp_lowres.pfm')) # load LF disparity map
    except:
        print(data_root+dir_LFimage+'/gt_disp_lowres.pfm..does not exist' % i )            
        
    return traindata, traindata_label


# Utils from HCI light field benchmark https://lightfield-analysis.uni-konstanz.de/tools
import configparser
def read_parameters(data_folder):
    params = dict()

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:
        parser = configparser.ConfigParser()
        parser.read_file(f)

        section = "intrinsics"
        params["width"] = int(parser.get(section, 'image_resolution_x_px'))
        params["height"] = int(parser.get(section, 'image_resolution_y_px'))
        params["focal_length_mm"] = float(parser.get(section, 'focal_length_mm'))
        params["sensor_size_mm"] = float(parser.get(section, 'sensor_size_mm'))
        params["fstop"] = float(parser.get(section, 'fstop'))

        section = "extrinsics"
        params["num_cams_x"] = int(parser.get(section, 'num_cams_x'))
        params["num_cams_y"] = int(parser.get(section, 'num_cams_y'))
        params["baseline_mm"] = float(parser.get(section, 'baseline_mm'))
        params["focus_distance_m"] = float(parser.get(section, 'focus_distance_m'))
        params["center_cam_x_m"] = float(parser.get(section, 'center_cam_x_m'))
        params["center_cam_y_m"] = float(parser.get(section, 'center_cam_y_m'))
        params["center_cam_z_m"] = float(parser.get(section, 'center_cam_z_m'))
        params["center_cam_rx_rad"] = float(parser.get(section, 'center_cam_rx_rad'))
        params["center_cam_ry_rad"] = float(parser.get(section, 'center_cam_ry_rad'))
        params["center_cam_rz_rad"] = float(parser.get(section, 'center_cam_rz_rad'))

        section = "meta"
        params["disp_min"] = float(parser.get(section, 'disp_min'))
        params["disp_max"] = float(parser.get(section, 'disp_max'))
        params["frustum_disp_min"] = float(parser.get(section, 'frustum_disp_min'))
        params["frustum_disp_max"] = float(parser.get(section, 'frustum_disp_max'))
        params["depth_map_scale"] = float(parser.get(section, 'depth_map_scale'))

        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params