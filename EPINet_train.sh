# python -W ignore main_EPINet.py --lr 1e-3 \
#                                  --dataset DDFF \
#                                  --val_frequency 100  \
#                                  --out_path logs/EPINet/DDFF_data

# python -W ignore main_EPINet.py --lr 1e-3 \
#                                  --dataset CVIA \
#                                  --val_frequency 100  \
#                                  --out_path logs/EPINet/CVIA_data                     
python -W ignore main_EPINet.py --lr 1e-4 \
                                 --dataset CVIA \
                                 --val_frequency 100  \
                                 --out_path logs/EPINet/CVIA_data_lr_1e-4