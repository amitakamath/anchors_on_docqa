# Run Anchors on a DocumentQA model
### Train a DocumentQA model
First things first, train a DocumentQA model on SQuAD data. (From CodaLab:) From /document-qa, run

```sh
$ export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES=0; 
$ python3 docqa/squad/build_squad_dataset.py --train_file train-v1.1.json --dev_file dev-v1.1.json
$ python3 docqa/scripts/ablate_squad.py confidence --num-epochs 25 --no-tfidf model
$ python3 docqa/eval/squad_eval.py -o pred.json -c dev model*
$ python eval_squad.py dev-v1.1.json pred.json > eval.json
```
You need to run this code to get the squad data for this task.

Replace the path in anchor/notebook/anchor_for_text.py with the path to your trained model:
```
model=load_model('/u/scr/kamatha/document-qa/model-1105-063301')
```

### Run Anchors Code
To run anchors code:
1. Activate virtualenv 
```sh
$ source activate py36-amita
```
2. Make your GPUs visible:
```sh
$ export CUDA_VISIBLE_DEVICES=0
```
3. Modify your PYTHONPATH:
```sh
$ export PYTHONPATH=${PYTHONPATH}:`pwd`/document-qa
$ export PYTHONPATH=${PYTHONPATH}:`pwd`/anchor
```
4. Run the code:
```sh
$ python anchors/notebooks/anchor_for_text.py
```
