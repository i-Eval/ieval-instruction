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
import glob
from tqdm import tqdm
import time
from datasets import Dataset



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_config_path", type=str, default="../configs/alpaca.yaml", help="path to the llm config file")
    parser.add_argument("--dataset_json_path", type=str, default="../data/mmlu/*.json", help="path to the dataset json file, enable glob pattern")
    parser.add_argument("--log_dir", type=str, default="../results", help="path to the log directory")
    return parser.parse_args()

def load_dataset_and_batching(dataset,batch_size,few_shots=0):
    batched_dataset = []

    for i in range(0, len(dataset['instances']), batch_size):
        current_batch = []
        # add example json to batch, each example has 3 keys: instruction, input, output, few_shot_instance if few_shots > 0
        for j in range(i, i+batch_size):
            if j >= len(dataset['instances']):
                break
            example = {
                "instruction": dataset['instruction'],
                "input": dataset['instances'][j]['input'],
                "output": dataset['instances'][j]['output'],
                "few_shot_instances": dataset['few_shot_instances'][:few_shots] if few_shots > 0 else []
            }
            current_batch.append(example)
        batched_dataset.append(current_batch)
    return batched_dataset

#HACK use prompt template to generate prompt
def load_dataset_to_hf_dataset(dataset,few_shots=0):
    hf_dataset = {"instruction": [], "input": [], "output": [], "few_shot_instances": []}
    for i in range(len(dataset['instances'])):
        hf_dataset["instruction"].append(dataset['instruction'])
        hf_dataset["input"].append(dataset['instances'][i]['input'])
        hf_dataset["output"].append(dataset['instances'][i]['output'])
        hf_dataset["few_shot_instances"].append(dataset['few_shot_instances'][:few_shots] if few_shots > 0 else [])
    return Dataset.from_dict(result)

def matching_start_with(gold_anwser,predicted_anwser):
    gold_answer_correct=' '+gold_anwser+'.'
    return gold_answer_correct in predicted_anwser or predicted_anwser.startswith(gold_anwser)
    # return predicted_anwser.startswith(gold_anwser)

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

    # load llm and prompt template 
    with open(args.llm_config_path, "r") as f:
        llm_config = yaml.safe_load(f)
    
    model_name = llm_config["model_name"]
    generator = load(model_path=llm_config["model_path"], batch_size=llm_config["batch_size"], device_map="auto")
    PROMPT_DICT = llm_config["zero_shot_template"]


    # load all data json file to a dict
    dataset = {}
    for data_file in glob.glob(args.dataset_json_path):
        with open(data_file) as f:
            data = json.load(f)
            dataset[data['task_name']] = data



    
    
    # loop over all tasks, and compute acc for each task.
    eval_results = {}
    
    


    for task_name, task_data in dataset.items():
        print(f"Evaluating Task: {task_name}")
        batched_dataset = load_dataset_and_batching(task_data, batch_size=llm_config["batch_size"], few_shots=llm_config["few_shots"])
        eval_results[task_name] = {}
        gold_labels = []
        pred_labels = []
        for batch in tqdm(batched_dataset):
            if len(batch[0]["input"]) != 0:
                final_prompts = [ PROMPT_DICT['with_input'].format_map({"instruction":i["instruction"],"input":i["input"]}) for i in batch]
            else:
                final_prompts = [PROMPT_DICT['no_input'].format_map({"instruction":i["instruction"]}) for i in batch]
            
            labels = [i["output"] for i in batch]
            gold_labels.extend(labels)

            results = generator(
                final_prompts,
                **llm_config["generate_config"],
            )
            # get the new generated text using the index of prompt
            for full_output, prompt in zip(results, final_prompts):
                pred_label = full_output[0]["generated_text"].split(prompt)[-1]
                pred_labels.append(pred_label)
        # compute acc
        acc = sum([matching_start_with(gold, pred) for gold, pred in zip(gold_labels, pred_labels)]) / len(gold_labels)
        eval_results[task_name]["acc"] = acc

        # log current eval log
        datetime = time.strftime("%Y%m%d-%H%M%S")
        with open(f"{args.log_dir}/{datetime}_ieval_log_{task_name}_{model_name}.txt", "w") as f:
            for answer,gold_answer in zip(pred_labels,gold_labels):
                f.write(f"predicted:\t{answer} | gold:\t{gold_answer}\n")
            f.write(f"Accuracy: {acc}")
        
        print(f"Accuracy: {acc}")
    
    # log all eval log
    datetime = time.strftime("%Y%m%d-%H%M%S")
    print(f"Saving all eval results to {args.log_dir}/{datetime}_ieval_log_{model_name}_all.json")

    # compute average acc
    eval_results["average_acc"] = sum([v["acc"] for v in eval_results.values()]) / len(eval_results)
    with open(f"{args.log_dir}/{datetime}_ieval_log_{model_name}_all.json", "w") as f:
        json.dump(eval_results, f)
    print(f"Average Accuracy: {eval_results['average_acc']}")





