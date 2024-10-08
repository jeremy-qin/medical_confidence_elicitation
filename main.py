default_params = {
    "dataset": "medqa",
    "model": "qwen",
    "sample_size": 500,
    "k": 1,
    "prompt_template": "vanilla",
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

import re
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from collections import Counter
from tqdm import tqdm
from langchain.chains import LLMChain

from prompt_templates import base_prompt_template, base_prompt_template_2, cot_prompt_template, atypical_prompt_template, atypical_situation_prompt_template

# Function to standardize and extract confidence score
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

def extract_diagnostic_table(answer_text):
    diagnostic_data = {
        'Diagnostic Hypotheses': [],
        'Findings in Favor': [],
        'Findings Against': [],
        'Expected Findings Not Described': [],
        'Likelihood': []
    }
    
    hypothesis_pattern = re.compile(r'\d+\.\s+(.+?)(?=\d+\.|\Z)', re.DOTALL)
    favor_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    against_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    not_described_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    likelihood_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    
    hypotheses = hypothesis_pattern.findall(answer_text.split('Diagnostic hypotheses:')[1].split('Findings that speak in favor of this hypothesis:')[0])
    findings_in_favor = favor_pattern.findall(answer_text.split('Findings that speak in favor of this hypothesis:')[1].split('Findings that speak against this hypothesis:')[0])
    findings_against = against_pattern.findall(answer_text.split('Findings that speak against this hypothesis:')[1].split('Findings expected to be present but not described in the case:')[0])
    expected_not_described = not_described_pattern.findall(answer_text.split('Findings expected to be present but not described in the case:')[1].split('Likelihood:')[0])
    likelihoods = likelihood_pattern.findall(answer_text.split('Likelihood:')[1])
    
    def clean_extracted_strings(tuples):
        return [item.strip() for sublist in tuples for item in sublist[1:]]

    diagnostic_data['Diagnostic Hypotheses'] = [hypo.strip() for hypo in hypotheses]
    diagnostic_data['Findings in Favor'] = clean_extracted_strings(findings_in_favor)
    diagnostic_data['Findings Against'] = clean_extracted_strings(findings_against)
    diagnostic_data['Expected Findings Not Described'] = clean_extracted_strings(expected_not_described)
    diagnostic_data['Likelihood'] = clean_extracted_strings(likelihoods)
    
    table_data = pd.DataFrame.from_dict(diagnostic_data)
    
    return table_data

def standardize_and_extract_confidence_and_reasoning(answer_text):
    match = re.search(r'answer is ([A-D]):.*?(\d+)% confident', answer_text, re.IGNORECASE | re.DOTALL)

    if match:
        reasoning_start = match.start()
        reasoning = answer_text[:reasoning_start].strip()

        standardized_answer = match.group(1).upper()
        confidence_score = int(match.group(2))
    else:
        reasoning = answer_text.strip() 
        standardized_answer = 'Unknown'
        confidence_score = None

    return standardized_answer, confidence_score, reasoning

def compute_ground_truth_probability(correct_answer, model_answers):
    correct_count = sum([1 for model_ans in model_answers if model_ans[0] == correct_answer])
    return correct_count / len(model_answers)

def get_majority_answer(answers):
    if not answers:
        return 'Unknown'
    counter = Counter(answers)
    majority_answer, majority_count = counter.most_common(1)[0]  # Get the most common answer
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
    import torch
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
    import torch
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

#experiment
def experiment(params):
    from dotenv import load_dotenv
    import numpy as np
    import os
    import pandas as pd
    from data import MedQA, CommonsenseQA, MedMCQA, PubmedQA
    import random
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama.llms import OllamaLLM

    load_dotenv()

    os.environ.get('OPENAI_API_KEY')
    os.environ.get('GOOGLE_API_KEY')
    os.environ.get('ANTHROPIC_API_KEY')
    
    print("Start of Experiment")

    dataset = params["dataset"]
    model = params["model"]
    prompt_template = params["prompt_template"]
    sample_size = params["sample_size"]
    k = params["k"]
    sampling = params["sampling"]


    if sampling == "base" and k > 1:
        raise ValueError("k must be 1 when sampling is 'base'.")
    elif sampling != "base" and k <= 1:
        raise ValueError("k must be > 1")

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

    if sample_size == "all":
        dev_examples = dev_data
        dev_answers = dev_data_answers
    else:
        print(len(dev_data))
        sampled_indices = random.sample(range(len(dev_data)), sample_size)
        dev_examples = [dev_data[i] for i in sampled_indices]
        dev_answers = [dev_data_answers[i] for i in sampled_indices]

    if prompt_template == "vanilla":
        template = base_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "atypical":
        template = atypical_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "atypical-situation":
        template = atypical_situation_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "cot":
        template = cot_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "deliberate_reflection":
        template = base_prompt_template_2()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    else:
        raise ValueError("Invalid prompt template")
    
    if model == "gpt3":
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    elif model == "gpt-4":
        llm = ChatOpenAI(model_name="gpt-4")
    elif model == "gpt-4-turbo":
        llm = ChatOpenAI(model_name="gpt-4-turbo")
    elif model == "claude":
        llm = ChatAnthropic(model='claude-3-sonnet-20240229')
    elif model == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    elif model == "llama3":
        llm = OllamaLLM(model="llama3.1")
    elif model == "qwen":
        llm = OllamaLLM(model="qwen2.5:7b")
        
    llm_chain = LLMChain(
        prompt = prompt,
        llm = llm
    )

    if prompt_template == "vanilla" or prompt_template == "cot" or prompt_template == "deliberate_reflection" or prompt_template == "atypical" or prompt_template == "atypical-situation":
        print("Using base prompt template")
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
                raw_answer = llm_chain.run(question)
                standardized_answer, difficulty_score, confidence_score = standardize_and_extract_details(raw_answer)
                vanilla_confidence = confidence_score
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
                if _ == 0 and i<10:
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

            average_difficulty = sum(temp_difficulty_scores) / len(temp_difficulty_scores) if temp_difficulty_scores else None
            avg_conf = avg_confidence(temp_answers, temp_scores, temp_answers[0]) if temp_scores else None
            average_vanilla_confidence = avg_confidence(temp_answers, temp_vanilla_scores, temp_answers[0]) if temp_scores else None
            ground_truth_probability = compute_ground_truth_probability(correct_answer, temp_answers)
            majority_answer, consistency = get_majority_answer(temp_answers)
            final_prediction = get_predicted_answer(temp_answers, temp_scores)

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
        
        df.to_parquet(f"./results/{model}/{dataset}_{prompt_template}_{sampling}_{sample_size}.parquet")
    
    # Compute Accuracy
    print("Computing accuracy")
    correct = 0
    for i in range(len(df)):
        if df['Correct Answers'][i] == df['Majority Predicted Answer'][i]:
            correct += 1
    accuracy = correct / len(df)
    acc = np.array([accuracy])
    print(accuracy)
    np.save(f"./results/{dataset}_{model}_{prompt_template}_{sampling}_{sample_size}_{current_time}_acc.npy", acc)

    #Compute ece
    print("Computing ece")
    ece = compute_ece(df, sampling)
    print("Computing Brier")
    brier = brier_score(df, sampling)
    print(brier)

    print("Computing AUC")
    auc_score = compute_auc(df, sampling)
    print(f"AUC Score: {auc_score}")

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