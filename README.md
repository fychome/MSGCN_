# MSGCN
This is a PyTorch implementation of the paper: MSGCN: Long Distance and Multi-Direction Graph Convolution Network for Traffic Forecasting. 

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Preparation
### Traffic Flow datasets

Download PEMS05 dataset from [https://github.com/Davidham3/STSGCN](https://github.com/Davidham3/STSGCN). Uncompress them and move them to the data folder.

```

# PEMS05
python ./lib/generate_flow_data.py --output_dir=data/PEMS05 --traffic_npy_filename=data/pems5flow.npy

# PEMS-BAY
python ./lib/generate_flow_data.py --output_dir=data/PEMS03 --traffic_npy_filename=data/pems3flow.npy

```

### Traffic Speed datasets
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 

```

# METR-LA
python ./lib/generate_speed_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python ./lib/generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Model Training

* METR-LA

```
python train.py --config_filename=./config/msgcn_metrla.yaml
```
* PEMS-BAY

```
python train.py --config_filename=./config/msgcn_pemsbay.yaml
```
* PEMS03

```
python train.py --config_filename=./config/msgcn_pems03.yaml
```
* PEMS05

```
python train.py --config_filename=./config/msgcn_pems05.yaml
```

### Run the Pre-trained Model
set the epoch=100 in yaml.
* METR-LA

```
python test.py --config_filename=./config/msgcn_metrla.yaml --output_filename=data/msgcn_predictions_metrla.npz
```
* PEMS-BAY

```
python test.py --config_filename=./config/msgcn_pemsbay.yaml --output_filename=data/msgcn_predictions_pemsbay.npz
```
* PEMS03

```
python test.py --config_filename=./config/msgcn_pems03.yaml --output_filename=data/msgcn_predictions_pems03.npz
```
* PEMS05

```
python test.py --config_filename=./config/msgcn_pems05.yaml --output_filename=data/msgcn_predictions_pems05.npz
```

