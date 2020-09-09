# GAT-Stroke

Edge graph attention network for stroke classification.

### Requirements

all Linux distributions no earlier than Ubuntu 16.04

Python 3.5 or later

PyTorch 1.0

### Installing

pip install -r requirements.txt

### Data

Download data.tar.gz and edge.tar.gz from

https://drive.google.com/open?id=1-0tRJGNZGBWgmMJ6NIzUN8QY115MBLWO

and put the data in the same directory of the code.

### Usage

tar -zxvf data.tar.gz

tar -zxvf edge.tar.gz

### train
python main.py --data_dir=data --edge_dir=edge/time_space --model_path=checkpoint/model.th --edge_feature_attn --edge_update --num_classes=2 --mode=train

### eval
python main.py --data_dir=data --edge_dir=edge/time_space --model_path=checkpoint/model.th --edge_feature_attn --edge_update --num_classes=2 --mode=eval
