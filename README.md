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

## start eval

```
python ./api/ServingLocalAlpacaStyle.py --serving_config_path ./configs/llama-i.yaml
```

start a new terminal
```
python run.py --serving_config_path ./configs/alpaca.yaml --dataset_path /home/cl/i-Eval/ieval/data/ieval_mmlu_anatomy_noendanwser.json --llm_service_address http://127.0.0.1:8080/llm_serving/

```

## Results
| Models(Zero-Shot)              | mmlu_astronomy | mmlu_anatomy | mmlu_college_mathematics | mmlu_abstract_algebra |
|---------------------|----------------|--------------|--------------------------|-----------------------|
| alpaca              | 0.40           | 0.41         | 0.31                     | 0.10                  |
| llama-i             | 0.53           | 0.41         | 0.29                     | 0.27                  |
| ChatGPT(2023.04.10) | 0.79           | 0.71         | 0.30                     | 0.35                  |

