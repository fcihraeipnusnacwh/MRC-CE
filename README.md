# MRC-CE
Code for the paper "Large-scale Fine-grained Concept Extraction Based on Machine Reading Comprehension for Knowledge Graph Completion"

# Requirements
- tensorflow==2.0.0  or  tensorflow==1.13.1
- scikit-learn==0.22.2  
- h5py<3.0.0

# Dataset
sample dataset is release on raw_data/
## format
- Chinese data: abstract \t 0/1 strings for the start position \t 0/1 strings for the end position
```
袁希治 1946年生，湖北汉阳人。二级演员。1966年毕业于湖北省戏曲学校楚剧科演员专业。	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```
- English data: abstract \t start position \t answer length \t question
```
California is a state in the Pacific Region of the United States of America. With 39.5 million residents across a total area of about 163,696 square miles 423,970 km2, California is the most populous U.S. state and the third-largest by area, and is also the world's thirty-fourth most populous subnational entity. California is also the most populated subnational entity in North America, and has its state capital in Sacramento. The Greater Los Angeles area and the San Francisco Bay Area are the nation's second- and fifth-most populous urban regions, with 18.7 million and 9.7 million residents respectively. Los Angeles is California's most populous city, and the country's second-most populous, after New York City. California also has the nation's most populous county, Los Angeles County, and its largest county by area, San Bernardino County. The City and County of San Francisco is both the country's second most densely populated major city after New York City and the fifth most densely populated county, behind only four of the five New York City boroughs.	654	4	city	what is the calif. concept?
```

# Run on sample data
## Run on tensorflow==1.13.1
- run on Chinese data
1. BERT-MRC
```
python run_squad.py \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --do_train=True \
  --train_file=raw_data/train.txt \
  --do_predict=True \
  --predict_file=raw_data/test.txt \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --output_dir=output/
```
2. Random Forest
First, you need to put the nbest_predictions.json and forest_construction.py in the same folder
```
python3 forest_construction.py
```
3. Rule_based_Pruning
```
python3 Rule_based_Pruning.py
```
- run on English data
```
python run_squad.py \
  --vocab_file=pretrained_model/cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=pretrained_model/cased_L-12_H-768_A-12/bert_model.ckpt \
  --train_file=raw_data/train_mrc.txt \
  --do_predict=True \
  --predict_file=raw_data/test_mrc.txt \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=4.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --do_lower_case=False \
  --output_dir=output/
```
2. Random Forest
First, you need to put the nbest_predictions.json and forest_construction.py in the same folder
```
python3 forest_construction.py
```
3. Rule_based_Pruning
```
python3 Rule_based_Pruning.py
```
## Run on tensorflow==2.0.0
Only release the code for Chinese data
```
train: ysy_find_concepts.py test
test: ysy_find_concepts.py train
```
# results
the results which are labeled by the volunteers are release on results/

# Reference
- google-research bert: <https://github.com/google-research/bert>
