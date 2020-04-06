# "A Deep Supervised Edge Optimization Algorithm for Salt Body Segmentation" code
This is the source code for our paper "A Deep Supervised Edge Optimization Algorithm for Salt Body Segmentation".



## Recent Update

**`2020.04.06`**: First push the code.


## Dependencies
- glob2==0.6
- Pillow==5.2.0
- opencv-python==3.4.0.12
- scikit-image==0.14.0
- scikit-learn==0.19.2
- scipy==1.1.0
- torch==0.3.1
- torchvision==0.2.0
- tensorboard==1.12.0
- tensorboardX==1.4





## Data Setup
### TGS train dataset
1.Download at
``` 
https://1drv.ms/u/s!AsLbpD-adIBLrHkuScdID781sUTr 
``` 
2.Unzip it under folder "tgs-dataset"

##Training
1.Go to the experiment folder
``` 
cd config/res34_loss_naive_nodice_hp_scse_ds_border
``` 
 
2.Start train model_34 fold 0 with batch-size 32：
```
sh train.sh 32
```

3.The code will automatically generate a "train_log" folder, each experiment result is in a folder with the same name as the experiment folder


## Reference
- https://arxiv.org/abs/1608.03983 LR schedule
- https://arxiv.org/abs/1803.02579 Squeeze and excitation
- https://arxiv.org/abs/1411.5752 Hypercolumns
- https://arxiv.org/abs/1705.08790 Lovasz loss
- https://www.kaggle.com/c/tgs-salt-identification-challenge tgs-salt-identification-challenge













