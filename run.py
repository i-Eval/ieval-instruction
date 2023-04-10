import json
import requests
import argparse
from tqdm import tqdm
import yaml

def request_anwser(address,examples):
    url = address
    payload = json.dumps({
        "instructions": [i['instruction'] for i in examples],
        "inputs": [i['input'] for i in examples],
        "few_shot_instances": [i['few_shot_instances'] for i in examples] ,
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()['answers']

def load_dataset_and_batching(dataset_path,batch_size,few_shots=0):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    task_name = dataset['task_name']

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
    return task_name,batched_dataset

def request_all_answers_gold_pair(address,dataset):
    all_answers = []
    all_gold_answers = []
    # tqdm as a progress bar
    for examples in tqdm(dataset):
        answers = request_anwser(address,examples)
        all_answers.append(answers)

        gold_answers = [example["output"] for example in examples]
        all_gold_answers.append(gold_answers)
    return all_answers,all_gold_answers



def matching_start_with(gold_anwser,predicted_anwser):
    return predicted_anwser.startswith(gold_anwser)
    

if __name__=="__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving_config_path", type=str, default="./configs/machiko.yaml")
    parser.add_argument("--dataset_path", type=str, default="./data/ieval_mmlu_anatomy.json")
    parser.add_argument("--llm_service_address", type=str, default="http://127.0.0.1:8080/llm_serving/")
    args = parser.parse_args()

    # get batch size in serving config
    with open(args.serving_config_path, "r") as f:
        serving_config = yaml.load(f, Loader=yaml.FullLoader)
    batch_size = serving_config['batch_size']
    model_name = serving_config['model_name']

    # load dataset
    task_name, dataset = load_dataset_and_batching(args.dataset_path, batch_size=batch_size, few_shots=0)

    # request all answers
    all_answers,all_gold_answers = request_all_answers_gold_pair(args.llm_service_address,dataset)

    # calculate accuracy
    accuracy = 0
    total_number = 0
    for answers,gold_answers in zip(all_answers,all_gold_answers):
        for answer,gold_answer in zip(answers,gold_answers):
            accuracy += matching_start_with(gold_answer,answer)
            total_number += 1
    accuracy /= total_number
    print("Accuracy: ",accuracy)

    # log modeloutput and accuracy
    with open(f"ieval_log_{task_name}_{model_name}.txt", "w") as f:
        for answers,gold_answers in zip(all_answers,all_gold_answers):
            for answer,gold_answer in zip(answers,gold_answers):
                f.write(f"predicted:\t{answer} | gold:\t{gold_answer}\n")
        f.write(f"Accuracy: {accuracy}")


