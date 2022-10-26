# Learning-Motion-Robust-Remote-Photoplethysmography-through-Arbitrary-Resolution-Videos
The code for paper __Learning Motion-Robust Remote Photoplethysmography through Arbitrary Resolution Videos__

We propose two plug-andplay blocks (i.e., physiological signal feature extraction block
(PFE) and temporal face alignment block (TFA)) to alleviate
the degradation of changing distance and head motion. 

## Environment
```
pip install pip_list.txt
```
## Before running

### Weights setting
```
Change the absolute path of the spynet_.pth in ./model/PhysNet_PFE_TFA.py line 106
The spynet_.pth is in the ./weights
```
### Dataloader setting
Load the dataset
```
Change the path to the dataset in ./dataloader/LoadVideotrain_pure.py line 91
```
Load the arbitrary_resolution_data
```
Change the path to the path_to_arbitrary_resolution_data in ./dataloader/LoadVideotrain_pure.py line 92
```
## Model training&testing
### Train the model
```
python train.py --gpu 0 --version 0
```

### Test the model in a 128x128 resolution.
```
python test.py --gpu 0 --version 0 --scale 1.0
```

## Dataset
The information of the PURE dataset please refer:
> Ronny Stricker, Steffen Muller, and Horst-Michael Gross. Non-contact video-based pulse rate measurement on a mobile service robot. In The 23rd IEEE International Symposium on Robot and Human Interactive Communication, pages 1056–1062. IEEE, 2014.
### Generate the arbitrary resolution frames
```
cd dataset
python data_deal_pure.py
```

The implementation of model PhysNet_PFE_TFA, PhysNet_PFE, Physformer_PFE_TFA_crcloss, Physformer_PFE_TFA, Physformer_PFE, and the dataloader/preprocess of UBFC-rppg, COHFACE, UBFC-phys, UCLA-rppg will be submitted to Github.