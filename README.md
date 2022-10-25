# Learning-Motion-Robust-Remote-Photoplethysmography-through-Arbitrary-Resolution-Videos
The code for paper <Learning Motion-Robust Remote Photoplethysmography through Arbitrary Resolution Videos>

## Environment
```
pip install pip_list.txt
```
## Model training&testing
Before running
```
Change the absolute path of the spynet_ in ./model/PhysNet_PFE_TFA.py line 106
```
For training the model
```
python train.py --gpu --version 
```

For testing the model in a 128x128 resolution.
```
python test.py --gpu --version --scale 1.0
```

## Dataset
Prepare the dataset
```
cd dataset
python data_deal_pure.py
```
Load the dataset
```
Change the path to the dataset in ./dataloader/LoadVideotrain_pure.py line 104
```
Load the arbitrary_resolution_data
```
Change the path to the path_to_arbitrary_resolution_data in ./dataloader/LoadVideotrain_pure.py line 105
```