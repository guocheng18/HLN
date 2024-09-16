# Session-based Recommendation with Hierarchical Leaping Networks

## Directory Structure
- datasets/
    - api.py (dataset interface)
    - lastfm.py (preprocess LastFM dataset)
    - yoochoose.py (preprocess YooChoose dataset)
- hln.py (model definition)
- main.py (train and evaluate)

## Get Started
### 1. Download datasets
- YooChoose: https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015
- LastFM: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

### 2. Preprocess datasets
```bash
mkdir -p yoochoose/raw yoochoose/processed
# Place yoochoose-clicks.dat under yoochoose/raw 
python datasets/yoochoose.py
```

### 3. Train and evaluate HLN
```bash
python main.py
```

## Citation
If you use `HLN`, please cite our work: 
```
@inproceedings{10.1145/3397271.3401217,
author = {Guo, Cheng and Zhang, Mengfei and Fang, Jinyun and Jin, Jiaqi and Pan, Mao},
title = {Session-based Recommendation with Hierarchical Leaping Networks},
year = {2020},
isbn = {9781450380164},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3397271.3401217},
booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
}
```