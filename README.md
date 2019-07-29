# KAKAO ARENA : 브런치 사용자를 위한 글 추천 대회  

### 0. Dev dataset performance  
MAP     0.055358  
NDCG    0.143128  
Entropy 7.996152  


### 1. Requirements  
python 3.6.8  
numpy 1.16.3  
tqdm 4.32.1  
pytorch 1.1.0  

### 2. Run  
#### a. Preprocess  
Configure ```dataset_path``` and ```prepro_path``` in preprocess file.  
Run ```Aren_prepro.ipynb``` or run ```python Aren_prepro.py```  
#### b. Data Build
```python data_builder.py --prepro_root PREPRO_PATH```
#### c. Train
```python main.py --prepro_root PREPRO_PATH```  
#### d. Test
Dev dataset for public leaderboard  
```python test.py --dataset_root DATA_PATH --prepro_root PREPRO_PATH --mode DEV --test_epoch TEST_EPOCH```  
Test dataset for private leaderboard  
```python test.py --dataset_root DATA_PATH --prepro_root PREPRO_PATH --mode TEST --test_epoch TEST_EPOCH```  
```TEST_EPOCH``` is epoch when best valid performance in train.  
results is in ```./``` and saved models is in ```--save_path SAVE_PATH```  


### 3. Model
2-layer의 GRU + Multi-head Attention Model  

GRU의 인풋은 사용자가 구독한 아이템 임베딩 벡터로, 메타데이터만 사용 (매거진과 컨텐츠 정보 미사용).  
아이템 임베딩 벡터는 writer_id, keyword_ids(5개), rg_ts_embedding, magazine_id 정보를 활용.  
각 id는 각 feature의 lookup table을 이용해 벡터화.
rg_ts_embedding은 rg_ts의 최소값과 최대값에 해당하는 학습 가능한 두 벡터를 설정 후, 선형보간으로 정의.  
인풋과 타겟은 사용자 히스토리의 윈도우 슬라이딩 형태로 정의했으며, 윈도우 사이즈가 N일 경우 앞 N-1개가 인풋, 마지막 아이템이 타겟으로 설정하였다.    

GRU의 아웃풋을 이용해 Multi-head Attention Model의 key, value로 입력.
Attention model의 query로는 사용자 임베딩 벡터이다.  
사용자 임베딩 벡터는 user_id, keyword_ids(8개), following_writer_id(8개), read_ts_embedding 정보를 활용.  

Attention의 결과값을 모든 아이템 벡터(아이템 정보들을 단순 FC)들과 곱. 상위 100개를 추천.
