# LTPHM: Long-term Traffic Prediction based on Hybrid Model

This is the original pytorch implementation of LTPHM in the following paper: 
[LTPHM: Long-term Traffic Prediction based on Hybrid Model, CIKM 2021].


<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Process raw data 

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Train Commands

```
python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
python train.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS08 --adjdata  /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems08.pkl --num_nodes 170 |tee pems08.txt

python train.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS08_speed --adjdata  /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems08.pkl --num_nodes 170 |tee pems08_speed.txt

conda activate cuda11
python train.py --device cuda:1 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS03 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems03.pkl --num_nodes 358 |tee pems03.txt

python train.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS03 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems03.pkl --num_nodes 358 |tee pems03drop_0.5.txt

conda activate cuda11
python train.py --device cuda:1 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS04 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems04.pkl --num_nodes 307 |tee pems04.txt

python train.py --device cuda:3 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS07 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems07.pkl --num_nodes 883 --batch_size 32 |tee pems07.txt

python train.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS07_speed --adjdata  /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems07.pkl --num_nodes 883 --batch_size 32 |tee pems07_speed.txt

python train.py --device cuda:1 --data /data/home2/huangchuyin/Graph-WaveNet/data/METR-LA --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx.pkl --num_nodes 207 |tee metr_la.txt

python train.py --device cuda:2 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS-BAY --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_bay.pkl --num_nodes 325 |tee bay.txt

python train.py --device cuda:2 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS04_speed --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems04.pkl --num_nodes 307 |tee pems04_speed.txt
```
Test
```
python test.py --device cuda:1 --data /data/home2/huangchuyin/Graph-WaveNet/data/METR-LA --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx.pkl --num_nodes 207  --checkpoint /data/home2/huangchuyin/GraphwaveNet_DCN/garage/version1-PEMS-BAY/metr_epoch_100_2.88.pth


python test.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS03 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems03.pkl --num_nodes 358 --checkpoint /data/home2/huangchuyin/GraphwaveNet_DCN/garage/version1-PEMS-BAY/metr_epoch_100_14.85.

python test.py --device cuda:0 --data /data/home2/huangchuyin/Graph-WaveNet/data/PEMS03 --adjdata /data/home2/huangchuyin/Graph-WaveNet/data/sensor_graph/adj_mx_pems03.pkl --num_nodes 358 --checkpoint /data/home2/huangchuyin/GraphwaveNet_DCN/garage/version1-PEMS-BAY/metr_epoch_100_14.85.pth
