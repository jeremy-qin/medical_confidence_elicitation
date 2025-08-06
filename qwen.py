import re
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import argparse
from datetime import datetime
import random
import os

# Import your existing modules
from prompt_templates import base_prompt_template, base_prompt_template_2, cot_prompt_template, atypical_prompt_template, atypical_situation_prompt_template

default_params = {
    "dataset": "medqa",
    "model": "qwen3-8b",
    "sample_size": "all",
    "k": 1,
    "prompt_template": "atypical-situation",
    "sampling": "base"
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

# Keep all your existing parsing functions
def standardize_and_extract_confidence(answer):
    match = re.search(r'([A-D]):\s*(.+?)\s*(\d+)%', answer, re.IGNORECASE)
    if match:
        standardized_answer = f"{match.group(1).upper()}"
        confidence_score = int(match.group(3))
    else:
        standardized_answer = answer[0] if answer else 'Unknown' 
        confidence_score = None
    return standardized_answer, confidence_score

def parse_full_output_and_store_scores(full_output_text):
    sections = full_output_text.strip().split('\n\n')
    scores = []
    pattern = re.compile(r':?\s*(?:\[\s*|\(\s*)?(\d+(?:\.\d+)?)(?:\s*\]|\s*\))?\s*$')
    
    for section in sections:
        if section.startswith("Symptoms and signs:"):
            lines = section.split('\n')
            for line in lines[1:]: 
                match = pattern.search(line)
                if match:
                    score = float(match.group(1))
                    scores.append(score)

    scores_array = np.array(scores)
    return scores_array

def parse_atypical_situation_scores(full_output_text):
    match = re.search(r'Atypicality: \[?(\d(?:\.\d+)?)\]?', full_output_text)
    if match:
        score = float(match.group(1))
    else:
        score = None
    
    return np.array([score]) if score is not None else np.array([])

def standardize_and_extract_details(answer_text):
    answer_match = re.search(r'- Answer(?: \(letter\))?: ([A-E])', answer_text, re.IGNORECASE)
    difficulty_match = re.search(r'- Difficulty: (\d+)', answer_text)
    confidence_match = re.search(r'- Confidence: (\d+)%', answer_text)

    if answer_match and difficulty_match and confidence_match:
        standardized_answer = answer_match.group(1).upper() 
        difficulty_score = int(difficulty_match.group(1))
        confidence_score = int(confidence_match.group(1))
    else:
        standardized_answer = 'Unknown'
        difficulty_score = None
        confidence_score = 50

    return standardized_answer, difficulty_score, confidence_score

def compute_ground_truth_probability(correct_answer, model_answers):
    correct_count = sum([1 for model_ans in model_answers if model_ans[0] == correct_answer])
    return correct_count / len(model_answers)

def get_majority_answer(answers):
    if not answers:
        return 'Unknown'
    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]
    consistency = majority_count / len(answers) if len(answers) > 0 else 0.0
    return majority_answer, consistency

def get_predicted_answer(answers, confidences):
    if not answers or not confidences or len(answers) != len(confidences):
        raise ValueError("The lengths of answers and confidences must be the same and non-empty.")
    answer_confidence_pairs = zip(answers, confidences)
    predicted_answer, max_confidence = max(answer_confidence_pairs, key=lambda pair: pair[1])
    return predicted_answer

def avg_confidence(candidate_answers, candidate_confidences, given_answer):
    if len(candidate_answers) != len(candidate_confidences):
        raise ValueError("The lengths of candidate_answers and candidate_confidences must be the same.")
    numerator = sum(confidence for answer, confidence in zip(candidate_answers, candidate_confidences) if answer == given_answer)
    denominator = sum(candidate_confidences)
    if denominator == 0:
        return None
    avg_conf = numerator / denominator
    return avg_conf

def compute_auc(df, sampling):
    from torchmetrics import AUROC
    auc = AUROC(task='binary')
    df['Target'] = df['Correct Answers'] == df['Final Prediction']
    target = torch.tensor(df['Target'].values).int()

    if sampling == "base":
        conf = torch.tensor(df['All Confidence Scores'].apply(lambda x: x[0]).values)/100
    elif sampling == "consistency":
        conf = torch.tensor(df['Consistency Confidence'].values)
    else:
        conf = torch.tensor(df['Average Confidence'].values)

    auroc = auc(conf, target)
    print(auroc)
    return auroc

def compute_ece(df, sampling):
    from torchmetrics import CalibrationError
    calibration_error = CalibrationError(n_bins=10, norm='l1', task='binary')
    df['Target'] = df['Correct Answers'] == df['Final Prediction']
    target = torch.tensor(df['Target'].values).int()

    if sampling == "base":
        conf = torch.tensor(df['All Confidence Scores'].apply(lambda x: x[0]).values)/100
    elif sampling == "consistency":
        conf = torch.tensor(df['Consistency Confidence'].values)
    else:
        conf = torch.tensor(df['Average Confidence'].values)

    ece = calibration_error(conf, target)
    print(ece)
    return ece

def brier_score(df, sampling):
    true_answers_col = "Correct Answers"
    predictions_col = "Final Prediction"

    if sampling == "average":
        confidence_col = "Average Confidence"
    elif sampling == "consistency":
        confidence_col = "Consistency Confidence"
    elif sampling == "base":
        confidence_col = "All Confidence Scores"

    true_answers = df[true_answers_col]
    predictions = df[predictions_col]

    if true_answers.empty or predictions.empty:
        raise ValueError("The DataFrame columns must not be empty.")

    outcomes = []
    confidences = []

    for index, row in df.iterrows():
        true_answer = row[true_answers_col]
        predicted_answer = row[predictions_col]
        if sampling == "base":
            confidence = row[confidence_col][0]/100
        else:
            confidence = row[confidence_col]

        if confidence > 1:
            confidence = 0.5
        outcome = 1 if predicted_answer == true_answer else 0

        outcomes.append(outcome)
        confidences.append(confidence)

    brier_score_sum = 0
    for conf, outcome in zip(confidences, outcomes):
        brier_score_sum += (conf - outcome) ** 2
    brier_score = brier_score_sum / len(df)

    return brier_score

class HuggingFaceModel:
    def __init__(self, model_name, quantization=None, max_new_tokens=512, temperature=0.1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        print(f"Loading {model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate quantization
        if quantization == "8bit":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Full precision
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate(self, prompt):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        response = full_response[len(prompt):].strip()
        return response

def get_model(model_name):
    """Initialize the appropriate model based on model name"""
    
    model_configs = {
        "qwen3-8b": ("/home/q/qinjerem/links/projects/aip-bangliu/qinjerem/models/Qwen3-8B", None),
        # "qwen2.5-7b-8bit": ("Qwen/Qwen2.5-7B-Instruct", "8bit"),
        # "qwen2.5-7b-4bit": ("Qwen/Qwen2.5-7B-Instruct", "4bit"),
        # "qwen2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", None),
        # "qwen2.5-14b-8bit": ("Qwen/Qwen2.5-14B-Instruct", "8bit"),
        # "qwen2.5-14b-4bit": ("Qwen/Qwen2.5-14B-Instruct", "4bit"),
        # "llama3-8b": ("meta-llama/Meta-Llama-3-8B-Instruct", None),
        # "llama3-8b-8bit": ("meta-llama/Meta-Llama-3-8B-Instruct", "8bit"),
        # "llama3-8b-4bit": ("meta-llama/Meta-Llama-3-8B-Instruct", "4bit"),
        # "llama3.1-8b": ("meta-llama/Meta-Llama-3.1-8B-Instruct", None),
        # "llama3.1-8b-8bit": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "8bit"),
        # "llama3.1-8b-4bit": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "4bit"),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_configs.keys())}")
    
    hf_model_name, quantization = model_configs[model_name]
    return HuggingFaceModel(hf_model_name, quantization)

def get_prompt_template(template_name):
    """Get the appropriate prompt template"""
    if template_name == "vanilla":
        return base_prompt_template()
    elif template_name == "atypical":
        return atypical_prompt_template()
    elif template_name == "atypical-situation":
        return atypical_situation_prompt_template()
    elif template_name == "cot":
        return cot_prompt_template()
    elif template_name == "deliberate_reflection":
        return base_prompt_template_2()
    else:
        raise ValueError(f"Invalid prompt template: {template_name}")

def experiment(params):
    from data import MedQA, CommonsenseQA, MedMCQA, PubmedQA
    
    print("Start of Experiment")
    
    dataset = params["dataset"]
    model_name = params["model"]
    prompt_template = params["prompt_template"]
    sample_size = params["sample_size"]
    k = params["k"]
    sampling = params["sampling"]

    if sampling == "base" and k > 1:
        raise ValueError("k must be 1 when sampling is 'base'.")
    elif sampling != "base" and k <= 1:
        raise ValueError("k must be > 1")

    # Load dataset
    if dataset == "medqa":
        data = MedQA("./datasets/medqa/data/")
        train_data = [x['question'] for x in data._train]
        dev_data = [x['question'] for x in data._dev]
        dev_data_answers = [x['answer'] for x in data._dev]
    elif dataset == "medmcqa":
        data = MedMCQA("./datasets/medmcqa/dev.json")
        dev_data = data._dev_questions
        dev_data_answers = data._dev_labels
    elif dataset == "pubmedqa":
        data = PubmedQA("./datasets/pubmedqa/dev.json")
        dev_data = data._dev_questions
        dev_data_answers = data._dev_labels
    elif dataset == "commonsenseqa":
        data = CommonsenseQA("./datasets/commonsenseqa/dev_rand_split.jsonl")
        dev_data = data._dev_questions
        dev_data_answers = data._dev_labels

    # Sample data
    if sample_size == "all":
        dev_examples = dev_data
        dev_answers = dev_data_answers
    else:
        print(f"Dataset size: {len(dev_data)}")
        sampled_indices = random.sample(range(len(dev_data)), sample_size)
        dev_examples = [dev_data[i] for i in sampled_indices]
        dev_answers = [dev_data_answers[i] for i in sampled_indices]

    # Get prompt template
    template = get_prompt_template(prompt_template)
    
    # Initialize model
    model = get_model(model_name)
    
    # Run experiment
    print(f"Running experiment with {len(dev_examples)} examples")
    
    answers_gpt = []
    consistency_scores = []
    vanilla_confidence_scores = []
    confidence_scores = []
    ground_truth_probabilities = []
    all_vanilla_confidence_scores = []
    all_confidence_scores = []
    all_predictions = []
    final_answer = []
    difficulty_scores = []
    mean_difficulty_scores = []
    atypical_scores_list = []

    i = 0

    for question, correct_answer in tqdm(zip(dev_examples, dev_answers), total=len(dev_examples)):
        temp_vanilla_scores = []
        temp_scores = []
        temp_answers = []
        temp_difficulty_scores = []

        for _ in range(k):
            # Format prompt with question
            formatted_prompt = template.replace("{question}", question)
            
            # Generate response
            raw_answer = model.generate(formatted_prompt)
            
            # Parse response
            standardized_answer, difficulty_score, confidence_score = standardize_and_extract_details(raw_answer)
            vanilla_confidence = confidence_score
            
            # Handle atypical scoring
            if prompt_template == "atypical":
                atypical_scores = parse_full_output_and_store_scores(raw_answer)
                print(f"Atypical Scores: {atypical_scores}")
                if len(atypical_scores) <= 0:
                    atypical_scores = np.array([1])
                calibrated_confidence = confidence_score * np.mean(np.exp(atypical_scores-1))
                print(f"Calibrated Confidence: {calibrated_confidence}")
                confidence_score = calibrated_confidence
            elif prompt_template == "atypical-situation":
                atypical_scores = parse_atypical_situation_scores(raw_answer)
                print(f"Atypical Scores: {atypical_scores}")
                if len(atypical_scores) <= 0:
                    atypical_scores = np.array([1])
                calibrated_confidence = confidence_score * np.mean(np.exp(atypical_scores-1))
                print(f"Calibrated Confidence: {calibrated_confidence}")
                confidence_score = calibrated_confidence
            else:
                atypical_scores = np.full(k, -999)

            temp_answers.append(standardized_answer)
            if difficulty_score is not None:
                temp_difficulty_scores.append(difficulty_score)
            if vanilla_confidence is not None:
                temp_vanilla_scores.append(vanilla_confidence)
            if confidence_score is not None:
                temp_scores.append(confidence_score)
                
            # Print debug info for first 10 examples
            if _ == 0 and i < 10:
                print(f"Question: {question}")
                print("\n")
                print(raw_answer)
                print("\n")
                print(f"Answer: {standardized_answer}")
                print(f"Difficulty: {difficulty_score}")
                print(f"Confidence: {vanilla_confidence}")
                print(f"Confidence_{prompt_template}: {confidence_score}")
                print("\n")
                print(f"Correct Answer: {correct_answer}")
                print("\n")

        i += 1

        # Compute aggregated metrics
        average_difficulty = sum(temp_difficulty_scores) / len(temp_difficulty_scores) if temp_difficulty_scores else None
        avg_conf = avg_confidence(temp_answers, temp_scores, temp_answers[0]) if temp_scores else None
        average_vanilla_confidence = avg_confidence(temp_answers, temp_vanilla_scores, temp_answers[0]) if temp_vanilla_scores else None
        ground_truth_probability = compute_ground_truth_probability(correct_answer, temp_answers)
        majority_answer, consistency = get_majority_answer(temp_answers)
        final_prediction = get_predicted_answer(temp_answers, temp_scores) if temp_scores else temp_answers[0]

        answers_gpt.append(majority_answer) 
        consistency_scores.append(consistency)
        confidence_scores.append(avg_conf)
        vanilla_confidence_scores.append(average_vanilla_confidence)
        ground_truth_probabilities.append(ground_truth_probability)
        all_vanilla_confidence_scores.append(temp_vanilla_scores)
        all_confidence_scores.append(temp_scores)
        all_predictions.append(temp_answers)
        final_answer.append(final_prediction)
        difficulty_scores.append(temp_difficulty_scores)
        mean_difficulty_scores.append(average_difficulty)
        atypical_scores_list.append(atypical_scores)

    # Create DataFrame
    df = pd.DataFrame({
        'Questions': dev_examples,
        'Difficulty Score': difficulty_scores,
        'Mean Difficulty Score': mean_difficulty_scores,
        'Correct Answers': dev_answers,
        'Majority Predicted Answer': answers_gpt,
        'All Predicted Answers': all_predictions,
        'Final Prediction': final_answer,
        'Consistency Confidence': consistency_scores,
        'Average Vanilla Confidence': vanilla_confidence_scores,
        'Average Confidence': confidence_scores,
        'All Vanilla Confidence Scores': all_vanilla_confidence_scores,
        'All Confidence Scores': all_confidence_scores,
        'Ground Truth Probability': ground_truth_probabilities,
        'Atypical Scores': atypical_scores_list
    })
    
    # Save results
    os.makedirs(f"./results/{model_name}", exist_ok=True)
    df.to_parquet(f"./results/{model_name}/{dataset}_{prompt_template}_{sampling}_{sample_size}.parquet")
    
    # Compute metrics
    print("Computing accuracy")
    correct = 0
    for i in range(len(df)):
        if df['Correct Answers'][i] == df['Majority Predicted Answer'][i]:
            correct += 1
    accuracy = correct / len(df)
    acc = np.array([accuracy])
    print(f"Accuracy: {accuracy}")
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    np.save(f"./results/{dataset}_{model_name}_{prompt_template}_{sampling}_{sample_size}_{current_time}_acc.npy", acc)

    print("Computing ECE")
    ece = compute_ece(df, sampling)
    
    print("Computing Brier Score")
    brier = brier_score(df, sampling)
    print(f"Brier Score: {brier}")

    print("Computing AUC")
    auc_score = compute_auc(df, sampling)
    print(f"AUC Score: {auc_score}")

    return df

if __name__ == "__main__":
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
    
    print("Parameters:", params)
    experiment(params)