default_params = {
    "dataset": "medqa",
    "model": "gpt-3.5-turbo",
    "sample_size": 100,
    "prompt_template": "base"
}

def defaults(dictionary, dictionary_defaults):
    for key, value in dictionary_defaults.items():
        if key not in dictionary:
            dictionary[key] = value
        else:
            if isinstance(value, dict) and isinstance(dictionary[key], dict):
                dictionary[key] = defaults(dictionary[key], value)
            elif isinstance(value, dict) or isinstance(dictionary[key], dict):
                raise ValueError("Given dictionaries have incompatible structure")
    return dictionary

import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from prompt_templates import base_prompt_template, cot_prompt_template

# Function to standardize and extract confidence score
def standardize_and_extract_confidence(answer):
    match = re.search(r'([A-D]):\s*(.+?)\s*(\d+)%', answer, re.IGNORECASE)
    if match:
        standardized_answer = f"{match.group(1).upper()}"
        confidence_score = int(match.group(3))
    else:
        standardized_answer = answer[0] if answer else 'Unknown'  # taking first character or marking as Unknown
        confidence_score = None
    return standardized_answer, confidence_score

# Function to standardize and extract confidence score
def standardize_and_extract_confidence_and_reasoning(answer_text):
    # Adjusted regex to handle additional text and variations in the answer format
    match = re.search(r'answer is ([A-D]):.*?(\d+)% confident', answer_text, re.IGNORECASE | re.DOTALL)

    if match:
        # Extracting reasoning (text before the matched sentence)
        reasoning_start = match.start()
        reasoning = answer_text[:reasoning_start].strip()

        # Extracting the answer letter and confidence score
        standardized_answer = match.group(1).upper()
        confidence_score = int(match.group(2))
    else:
        # Defaults if the expected format is not found
        reasoning = answer_text.strip()  # Consider entire text as reasoning
        standardized_answer = 'Unknown'
        confidence_score = None

    return standardized_answer, confidence_score, reasoning

# Function to compute ground truth probability based on the first character
def compute_ground_truth_probability(correct_answer, model_answers):
    correct_count = sum([1 for model_ans in model_answers if model_ans[0] == correct_answer])
    return correct_count / len(model_answers)

# Function to determine the majority answer
def get_majority_answer(answers):
    if not answers:
        return 'Unknown'
    counter = Counter(answers)
    majority_answer, _ = counter.most_common(1)[0]  # Get the most common answer
    return majority_answer


#experiment
def experiment(params):
    import numpy as np
    import pandas as pd
    from data import MedQA
    from langchain import PromptTemplate
    from langchain.chat_models import ChatOpenAI

    print("Start of Experiment")

    dataset = params["dataset"]
    model = params["model"]
    prompt_template = params["prompt_template"]
    sample_size = params["sample_size"]

    if dataset == "medqa":
        data = MedQA("./datasets/medqa/")

    train_data = [x['question'] for x in data._train]
    dev_data = [x['question'] for x in data._dev]
    dev_data_answers = [x['answer'] for x in data._dev]

    if sample_size == "all":
        dev_examples = dev_data
        dev_answers = dev_data_answers
    else:
        dev_examples = dev_data[:sample_size]
        dev_answers = dev_data_answers[:sample_size]

    if prompt_template == "base":
        template = base_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "cot":
        template = cot_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    else:
        raise ValueError("Invalid prompt template")
    
    if model == "gpt-3.5-turbo":
        gpt = ChatOpenAI(model_name='gpt-3.5-turbo')
        
    llm_chain = LLMChain(
        prompt = prompt,
        llm = gpt
    )

    if prompt_template == "base":
        print("Using base prompt template")
        answers_gpt = []
        confidence_scores = []
        ground_truth_probabilities = []
        all_confidence_scores = []
        all_predictions = []

        for question, correct_answer in tqdm(zip(dev_examples, dev_answers), total=len(dev_examples)):
            temp_scores = []
            temp_answers = []

            for _ in range(5):
                raw_answer = llm_chain.run(question)
                standardized_answer, confidence_score = standardize_and_extract_confidence(raw_answer)
                temp_answers.append(standardized_answer)
                if confidence_score is not None:
                    temp_scores.append(confidence_score)

            average_confidence = sum(temp_scores) / len(temp_scores) if temp_scores else None
            ground_truth_probability = compute_ground_truth_probability(correct_answer, temp_answers)
            majority_answer = get_majority_answer(temp_answers)

            answers_gpt.append(majority_answer)  # Storing only the majority answer
            confidence_scores.append(average_confidence)
            ground_truth_probabilities.append(ground_truth_probability)
            all_confidence_scores.append(temp_scores)
            all_predictions.append(temp_answers)

        # Creating DataFrame
        df = pd.DataFrame({
            'Questions': dev_examples,
            'Correct Answers': dev_answers,
            'Majority Predicted Answer': answers_gpt,
            'All Predicted Answers': all_predictions,
            'Average Confidence': confidence_scores,
            'All Confidence Scores': all_confidence_scores,
            'Ground Truth Probability': ground_truth_probabilities
        })

        df.to_parquet(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}.parquet")
    
    elif prompt_template == "cot":
        print("Using cot prompt template")
        answers_gpt = []
        confidence_scores = []
        ground_truth_probabilities = []
        all_confidence_scores = []
        all_predictions = []
        reasonings = []

        for question, correct_answer in tqdm(zip(dev_examples, dev_answers), total=len(dev_examples)):
            temp_scores = []
            temp_answers = []
            temp_reasonings = []

            for i in range(3):
                raw_answer = llm_chain.run(question)
                standardized_answer, confidence_score, reasoning = standardize_and_extract_confidence_and_reasoning(raw_answer)
                temp_answers.append(standardized_answer)
                temp_reasonings.append(reasoning)
                if confidence_score is not None:
                    temp_scores.append(confidence_score)

            average_confidence = sum(temp_scores) / len(temp_scores) if temp_scores else None
            ground_truth_probability = compute_ground_truth_probability(correct_answer, temp_answers)
            majority_answer = get_majority_answer(temp_answers)

            answers_gpt.append(majority_answer)  # Storing only the majority answer
            confidence_scores.append(average_confidence)
            ground_truth_probabilities.append(ground_truth_probability)
            all_confidence_scores.append(temp_scores)
            all_predictions.append(temp_answers)
            reasonings.append(temp_reasonings[-1])

        # Creating DataFrame
        df = pd.DataFrame({
            'Questions': dev_examples,
            'Reasoning': reasonings,
            'Correct Answers': dev_answers,
            'Majority Predicted Answer': answers_gpt,
            'All Predicted Answers': all_predictions,
            'Average Confidence': confidence_scores,
            'All Confidence Scores': all_confidence_scores,
            'Ground Truth Probability': ground_truth_probabilities
        })
    
    # Compute accuracy
    correct = 0
    for i in range(len(df)):
        if df['Correct Answers'][i] == df['Majority Predicted Answer'][i]:
            correct += 1
    accuracy = correct / len(df)
    acc = np.array([accuracy])
    np.save(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}_acc.npy", acc)

    #compute ece
    df['Is Correct'] = df['Majority Predicted Answer'] == df['Correct Answers']
    bins = np.linspace(0, 100, 6)  # 5 bins from 0 to 1
    bin_indices = np.digitize(df['Average Confidence'], bins) - 1
    ece = 0
    for i in range(len(bins)-1):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_confidence = np.mean(df.loc[bin_mask, 'Average Confidence']) / 100
            bin_accuracy = np.mean(df.loc[bin_mask, 'Is Correct'])
            bin_proportion = np.sum(bin_mask) / len(df)
            ece += bin_proportion * abs(bin_accuracy - bin_confidence)
    ece = np.array([ece])
    np.save(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}_ece.npy", ece)

if __name__ == "__main__":
    import json
    import argparse

    from datetime import datetime

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", type=str, help="JSON params file")
    parser.add_argument("--direct", "-d", type=str, help="JSON state string")

    arguments = parser.parse_args()

    if arguments.direct is not None:
        params = json.loads(arguments.direct)
    elif arguments.params is not None:
        with open(arguments.params) as file:
            params = json.load(file)
    else:
        params = {}

    params = defaults(params, default_params)

    experiment(params)