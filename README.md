## environment
```
conda create -n ieval python=3.10
conda activate ieval

pip install -r requirements.txt
cd src/transformers
pip install -e .
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu116 # set correct cuda verison
cd ../..
```

## current supported tasks
1. MMLU (56 subjects)
2. TruthfulQA (mc1 , mc2)
3. HellaSwag (same as HELM test examples)

## start eval

1. set up llm service
```
python ./api/ServingLocalAlpacaStyle.py --serving_config_path ./configs/alpaca.yaml --host 0.0.0.0 --port 8080
```

2. start a new terminal, make evaluation request
```bash
bash eval_mmlu_all.sh # eval all mmlu task in zero shot manner
python compute_avg_acc_mmlu.py ./results alpaca mmlu # compute avg acc on all mmlu tasks
```




