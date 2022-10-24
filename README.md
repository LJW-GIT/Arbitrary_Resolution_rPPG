# Learning-Motion-Robust-Remote-Photoplethysmography-through-Arbitrary-Resolution-Videos
The code for paper <Learning Motion-Robust Remote Photoplethysmography through Arbitrary Resolution Videos>

## Environment
```
pip install pip_list.txt
```
## Model training&testing
For training the model
```
python train.py --gpu --version 
```

For testing the model in a 128x128 resolution.
```
python test.py --gpu --version --scale 1.0
```

## Dataset
Load the dataset
```
Change the path to the dataset in ./dataloader/LoadVideotrain_pure.py line 104
```
Load the arbitrary_resolution_data
```
Change the path to the path_to_arbitrary_resolution_data in ./dataloader/LoadVideotrain_pure.py line 105
```