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

1. set up llm service
```
python ./api/ServingLocalAlpacaStyle.py --serving_config_path ./configs/alpaca.yaml --host 0.0.0.0 --port 8080
```

2. start a new terminal, make evaluation request
```bash
bash eval_mmlu_all.sh # eval all mmlu task in zero shot manner
```

## Todos

1. add more tasks' data
2. speed up inference
3. add api for evalauting OpenAI models



## Results
| Models(Zero-Shot)              | mmlu_astronomy | mmlu_anatomy | mmlu_college_mathematics | mmlu_abstract_algebra |
|---------------------|----------------|--------------|--------------------------|-----------------------|
| alpaca              | 0.40           | 0.41         | 0.31                     | 0.10                  |
| llama-i             | 0.53           | 0.41         | 0.29                     | 0.27                  |
| ChatGPT(2023.04.10) | 0.79           | 0.71         | 0.30                     | 0.35                  |

