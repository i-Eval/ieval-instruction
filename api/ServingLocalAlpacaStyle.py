from transformers import pipeline
import torch
import json
from pathlib import Path
from typing import List
from fastapi import FastAPI
import uvicorn
import torch.distributed as dist
import yaml
from pydantic import BaseModel



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving_config_path", type=str, default="../configs/machiko.yaml")
    return parser.parse_args()


app = FastAPI()


def load(
    model_path: str,
    batch_size: int,
    **kwargs,
) -> pipeline:
    generator = pipeline(
        "text-generation",
        model=model_path,
        **kwargs,
    )
    return generator

if __name__ == "__main__":
    args = get_args()
    with open(args.serving_config_path, "r") as f:
        serving_config = yaml.safe_load(f)

    generator = load(model_path=serving_config["model_path"], batch_size=serving_config["batch_size"], device_map="auto")

    PROMPT_DICT = serving_config["zero_shot_template"]
    
    #HACK build directly from 
    class Config(BaseModel):
        instructions: List[str]
        inputs: List[str]
        few_shot_examples: List[str] = []
        do_sample: bool = serving_config['generate_config'].get('do_sample', True)
        max_new_tokens: int = serving_config['generate_config'].get('max_new_tokens', 512)
        temperature: float = serving_config['generate_config'].get('temperature', 0.7)
        top_p: float = serving_config['generate_config'].get('top_p', 0.9)
        top_k: int = serving_config['generate_config'].get('top_k', 50)


    @app.post("/llm_serving/")
    def generate(config: Config):
        if len(config.instructions) > serving_config["batch_size"]:
            return {"error": "too much prompts."}
        
        if len(config.inputs)!=0 and len(config.inputs) != len(config.instructions):
            return {"error": "inputs and instructions should be the same length or no inputs."}

        if len(config.inputs)!=0:
            final_prompts = [ PROMPT_DICT['with_input'].format_map({"instruction":ins,"input":inp}) for ins,inp in zip(config.instructions,config.inputs) ]
        else:
            final_prompts = [PROMPT_DICT['no_input'].format_map({"instruction":ins}) for ins in config.instructions]
        
        results = generator(
            final_prompts,
            do_sample=config.do_sample,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        return {"responses": results,"answers": [i[0]['generated_text'].split("Response:")[1] for i in results]}

    uvicorn.run(app, host="0.0.0.0", port=8181)
