# python -W ignore main_FSNet.py --lr 1e-3 \
#                             --dataset DDFF \
#                             --out_path logs/FSNet/DDFF_data/nF_7 \
#                             --disp_mult 0.25 \
#                             --offset 0.25 \
#                             --list_root DDFF_dataset/lists/FS_nF_7
# python -W ignore main_FSNet.py --lr 1e-4 \
#                             --dataset CVIA_down_2x \
#                             --out_path logs/FSNet/CVIA_data_down_2x/nF_7_lr_1e-4 \
#                             --disp_mult 0.7 \
#                             --offset 0.9 \
#                             --list_root CVIA_dataset/dataset03/lists/FS_nF_7
# python -W ignore main_FSNet.py --lr 1e-4 \
#                             --dataset CVIA \
#                             --out_path logs/trial \
#                             --disp_mult 0.7 \
#                             --offset 0.9 \
#                             --list_root CVIA_dataset/dataset03/lists/FS_nF_7

# python -W ignore main_FSNet.py --lr 1e-4 \
#                             --dataset DDFF_down_2x_except_last \
#                             --out_path logs/FSNet/DDFF_data_down_2x_except_last/nF_7_lr_1e-4 \
#                             --disp_mult 0.25 \
#                             --offset 0.25 \
#                             --list_root DDFF_dataset/lists/FS_nF_7

# python -W ignore main_FSNet.py --lr 1e-4 \
#                             --dataset CVIA_down_2x_except_last \
#                             --out_path logs/FSNet/CVIA_data_down_2x_except_last/nF_7_lr_1e-4 \
#                             --disp_mult 0.7 \
#                             --offset 0.9 \
#                             --list_root CVIA_dataset/dataset03/lists/FS_nF_7

# python -W ignore main_FSNet.py --lr 1e-4 \
#                             --dataset DDFF_blur \
#                             --out_path logs/FSNet/DDFF_data_blur_rate_16/nF_7_lr_1e-4 \
#                             --disp_mult 0.25 \
#                             --offset 0.25 \
#                             --list_root DDFF_dataset/lists/FS_nF_7 \
#                             --blur_rate 16
                              

python -W ignore main_FSNet.py --lr 1e-4 \
                            --dataset CVIA_blur \
                            --out_path logs/FSNet/CVIA_data_blur_rate_16/nF_7_lr_1e-4 \
                            --disp_mult 0.7 \
                            --offset 0.9 \
                            --list_root CVIA_dataset/dataset03/lists/FS_nF_7 \
                            --blur_rate 16