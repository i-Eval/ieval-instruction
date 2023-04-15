# ieval-instruction

This repo is for evaluating your Instruction-Tuned models on open benchmarks, with just one command. All datasets are transformed and evaluated through instruction tuning paradigm.


## Currently supported tasks
1. MMLU (56 subjects)
2. TruthfulQA (mc1 , mc2)
3. HellaSwag (same as HELM's test examples, without acc normalization)

All supported tasks are under `./data`, you can add new dataset following the same json format. Pull request for adding new dataset is also welcomed.

## Environment
```
conda create -n ieval python=3.10
conda activate ieval

pip install -r requirements.txt
cd src/transformers
pip install -e .
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu116 # set correct cuda verison
cd ../..
```

## Features

1. One command evaluation 

```bash
python ./api/run.py --llm_config_path=../configs/alpaca.yaml --dataset_json_path=../ieval/data/mmlu/ieval_mmlu_*.json --log_dir=../results/test

# --llm_config_path : llm config, generation config
# --dataset_json_path : dataset paths, support glob pattern matching
# --log_dir : dir to save the eval logs

```


2. LLM service

First, set up a llm service
```bash
python ./api/serve.py --serving_config_path ./configs/alpaca.yaml --host 0.0.0.0 --port 8080

# change serving_config_path to your own model config yaml file.
```

Second, make request to the llm service, you can following request format in `test_serve.py` to make requests

```bash
python ./api/test_serve.py --serving_config_path ./configs/alpaca.yaml --dataset_path ./data/mmlu/ieval_mmlu_college_biology.json --llm_service_address http://127.0.0.1:8080/llm_serving/
```


## Examples

1. MMLU
```bash
python ./api/run.py --llm_config_path=../configs/alpaca.yaml --dataset_json_path=../ieval/data/mmlu/ieval_mmlu_*.json --log_dir=../results/test
```



## Todos
1. enable few shots evaluation
2. faster inference on local machine
3. ieval run/interactive/serve client
4. pip install ieval-instruction


