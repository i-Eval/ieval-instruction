# ieval-instruction

This repo is for evaluating your Instruction-Tuned models on open benchmarks, with just one command. All datasets are tranformed and evaluated through instruction tuning paradigm.


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

## Start eval

1. set up llm service
```
python ./api/ServingLocalAlpacaStyle.py --serving_config_path ./configs/alpaca.yaml --host 0.0.0.0 --port 8080

# change serving_config_path to your own model config yaml file.
```

2. start a new terminal, make evaluation request, the result will be saved under ./results

```bash
python run.py --serving_config_path ./configs/alpaca.yaml --dataset_path ./data/mmlu/ieval_mmlu_college_biology.json --llm_service_address http://127.0.0.1:8080/llm_serving/

# python run.py --serving_config_path <serving_config_path> --dataset_path <ieval_dataset_json> --llm_service_address <llm_serving_address>
```


## Examples

1. MMLU
```bash
bash scripts/eval_mmlu_all.sh # eval all mmlu task in zero shot manner
python scripts/compute_avg_acc_mmlu.py ./results alpaca mmlu # compute avg acc on all mmlu tasks
```

2. TruthfulQA
```
python run.py --serving_config_path ./configs/alpaca.yaml --dataset_path ./data/truthful_qa/ieval_truthful_qa_mc1.json --llm_service_address http://127.0.0.1:8080/llm_serving/
```

3. OpenAI models

`./scripts/eval_oai.py` can be used to evaluate OPENAI's models.


## Todos
1. enable few shots evaluation
2. faster inference on local machine


