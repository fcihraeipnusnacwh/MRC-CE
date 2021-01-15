# Code For Baseline
1. bert-bilstm-crf
2. xlm-roberta
3. xlnet-NER
4. flair

# bert-bilstm-crf
- run on Chinese data
```
python3 run.py \ 
 -data_dir=dataset \  
 -output_dir=output \  
 -init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \  
 -bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \  
 -vocab_file=chinese_L-12_H-768_A-12/vocab.txt \  
 -num_train_epochs=3.0 \  
 --batch_size=8
```
- run on English data  
start the bert-serving
```
nohup python3 -u bert-serving-start -max_seq_len 250 -pooling_strategy NONE -pooling_layer -4 -3 -2 -1 -model_dir uncased_L-12_H-768_A-12 -num_worker 4 >ysy_bert.log 2>&1 &
```
train the data
```
python3 train.py
```

# xlm-roberta
- run on Chinese data
```
python main.py 
      --data_dir=data/ \
      --task_name=ner   \
      --output_dir=model_dir/   \
      --max_seq_length=16   \
      --num_train_epochs 3  \
      --do_eval \
      --warmup_proportion=0.1 \
      --pretrained_path pretrained_models/xlmr.base/ \
      --learning_rate 0.00007 \
      --do_train \
      --eval_on test \
      --train_batch_size 8
      -- dropout 0.2
```
- run on English data
```
python main.py 
      --data_dir=data/ \
      --task_name=ner   \
      --output_dir=model_dir/   \
      --max_seq_length=16   \
      --num_train_epochs 3  \
      --do_eval \
      --warmup_proportion=0.1 \
      --pretrained_path pretrained_models/xlmr.base/ \
      --learning_rate 0.00007 \
      --do_train \
      --eval_on test \
      --train_batch_size 8
      -- dropout 0.2
```

# xlnet-NER
- run on Chinese data
```
python3 model.py --entry train
```
```
python3 model.py --entry test
```
- run on English data
```
python3 model.py --entry train
```
```
python3 model.py --entry test
```

# flair
- run on Chinese data
```
python3 flair_baseline_ch.py
```
- run on English data
```
python3 flair_baseline_en.py
```
# Reference
- XLNet-NER: <https://github.com/Ma-Dan/XLNet-ChineseNER>
- Flair: <https://github.com/flairNLP/flair>
- xlm-roberta: <https://github.com/mohammadKhalifa/xlm-roberta-ner>
- bert-bilstm-crf: <https://github.com/macanv/BERT-BiLSTM-CRF-NER>
- bert-bilstm-crf: <https://github.com/kyzhouhzau/Bert-BiLSTM-CRF>
