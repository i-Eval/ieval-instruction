This repo is for evaluating your Instruction-Tuned models on open Benchmarks, with just one command.


## Currently supported tasks
1. MMLU (56 subjects)
2. TruthfulQA (mc1 , mc2)
3. HellaSwag (same as HELM's test examples, without acc normalization)

all supported tasks are under `./data`, you can add new dataset following the same json format

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

## Start eval

1. set up llm service
```
python ./api/ServingLocalAlpacaStyle.py --serving_config_path ./configs/alpaca.yaml --host 0.0.0.0 --port 8080
# change serving_config_path to your own model config yaml file.
```

2. start a new terminal, make evaluation request, the result will be saved under ./results

```bash
# python run.py --serving_config_path <serving_config_path> --dataset_path <ieval_dataset_json> --llm_service_address <llm_serving_address>
python run.py --serving_config_path ./configs/alpaca.yaml --dataset_path ./data/mmlu/ieval_mmlu_college_biology.json --llm_service_address http://127.0.0.1:8080/llm_serving/
```


## Examples

1. MMLU
```bash
bash eval_mmlu_all.sh # eval all mmlu task in zero shot manner
python compute_avg_acc_mmlu.py ./results alpaca mmlu # compute avg acc on all mmlu tasks
```

2. HellaSwag
```
python run.py --serving_config_path ./configs/alpaca.yaml --dataset_path ./data/hellaswag/ieval_hellaswag_helm_zs.json --llm_service_address http://127.0.0.1:8080/llm_serving/
```


## Todos
1. enable few shots evaluation
2. faster inference on local machine


