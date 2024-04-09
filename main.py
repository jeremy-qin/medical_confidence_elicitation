default_params = {
    "dataset": "commonsenseqa",
    "model": "gpt-3.5-turbo",
    "sample_size": 500,
    "k": 1,
    "prompt_template": "atypical"
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
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain

from prompt_templates import base_prompt_template, base_prompt_template_2, cot_prompt_template, atypical_prompt_template

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

def parse_full_output_and_store_scores(full_output_text):
    # Split the text into sections
    sections = full_output_text.strip().split('\n\n')
    
    # Initialize an empty list to store scores
    scores = []
    
    # Process only the "Symptoms and signs" section
    for section in sections:
        if section.startswith("Symptoms and signs:"):
            # Split the section into lines
            lines = section.split('\n')
            for line in lines:
                # Search for lines with scores using the updated regex
                match = re.search(r'- .*: (\d)', line)
                if match:
                    # Extract the score and convert it to an integer
                    score = int(match.group(1))
                    if score == None:
                        score = 1
                    scores.append(score)
                    
    # Convert the list of scores to a numpy array
    scores_array = np.array(scores)
    return scores_array

# def standardize_and_extract_details(answer_text):
#     # Adjusted regular expressions to match the LLM's actual answer format
#     answer_match = re.search(r'- Answer: ([A-D])', answer_text, re.IGNORECASE)
#     difficulty_match = re.search(r'- Difficulty: (\d+)', answer_text)
#     confidence_match = re.search(r'- Confidence: (\d+)%', answer_text)

#     if answer_match and difficulty_match and confidence_match:
#         standardized_answer = answer_match.group(1).upper()  # Ensuring uppercase for consistency
#         difficulty_score = int(difficulty_match.group(1))
#         confidence_score = int(confidence_match.group(1))
#     else:
#         # Defaults if the expected format is not found
#         standardized_answer = 'Unknown'
#         difficulty_score = None
#         confidence_score = None

#     # No reasoning extraction is defined in this snippet; add if needed
#     return standardized_answer, difficulty_score, confidence_score

def standardize_and_extract_details(answer_text):
    # Adjust the regular expression to make the "(letter)" part optional
    # and include 'E' as a valid answer choice
    answer_match = re.search(r'- Answer(?: \(letter\))?: ([A-E])', answer_text, re.IGNORECASE)
    difficulty_match = re.search(r'- Difficulty: (\d+)', answer_text)
    confidence_match = re.search(r'- Confidence: (\d+)%', answer_text)

    if answer_match and difficulty_match and confidence_match:
        standardized_answer = answer_match.group(1).upper()  # Ensuring uppercase for consistency
        difficulty_score = int(difficulty_match.group(1))
        confidence_score = int(confidence_match.group(1))
    else:
        # Defaults if the expected format is not found
        standardized_answer = 'Unknown'
        difficulty_score = None
        confidence_score = 50

    return standardized_answer, difficulty_score, confidence_score

def extract_diagnostic_table(answer_text):
    # Initialize a dictionary to hold the extracted data
    diagnostic_data = {
        'Diagnostic Hypotheses': [],
        'Findings in Favor': [],
        'Findings Against': [],
        'Expected Findings Not Described': [],
        'Likelihood': []
    }
    
    # Define patterns for matching the different sections of the diagnostic table
    hypothesis_pattern = re.compile(r'\d+\.\s+(.+?)(?=\d+\.|\Z)', re.DOTALL)
    favor_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    against_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    not_described_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    likelihood_pattern = re.compile(r'\d+\.\s+(.+?):(.+?)(?=\d+\.|\Z)', re.DOTALL)
    
    # Extract data for each section
    hypotheses = hypothesis_pattern.findall(answer_text.split('Diagnostic hypotheses:')[1].split('Findings that speak in favor of this hypothesis:')[0])
    findings_in_favor = favor_pattern.findall(answer_text.split('Findings that speak in favor of this hypothesis:')[1].split('Findings that speak against this hypothesis:')[0])
    findings_against = against_pattern.findall(answer_text.split('Findings that speak against this hypothesis:')[1].split('Findings expected to be present but not described in the case:')[0])
    expected_not_described = not_described_pattern.findall(answer_text.split('Findings expected to be present but not described in the case:')[1].split('Likelihood:')[0])
    likelihoods = likelihood_pattern.findall(answer_text.split('Likelihood:')[1])
    
    # Function to clean up extracted strings
    def clean_extracted_strings(tuples):
        return [item.strip() for sublist in tuples for item in sublist[1:]]

    # Populate the dictionary with cleaned data
    diagnostic_data['Diagnostic Hypotheses'] = [hypo.strip() for hypo in hypotheses]
    diagnostic_data['Findings in Favor'] = clean_extracted_strings(findings_in_favor)
    diagnostic_data['Findings Against'] = clean_extracted_strings(findings_against)
    diagnostic_data['Expected Findings Not Described'] = clean_extracted_strings(expected_not_described)
    diagnostic_data['Likelihood'] = clean_extracted_strings(likelihoods)
    
    # Create a DataFrame from the dictionary
    table_data = pd.DataFrame.from_dict(diagnostic_data)
    
    return table_data

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

def compute_auc(df):
    # Convert confidence scores to a scale of 0 to 1
    df['Average Confidence'] /= 100

    # Sort DataFrame by confidence in descending order for cumulative operations
    df_sorted = df.sort_values(by='Average Confidence', ascending=False)

    # Initialize lists to store coverage and selective accuracy values
    coverage_values = []
    selective_accuracy_values = []

    # Define thresholds
    thresholds = np.linspace(0, 1, 21)  # Example: 21 thresholds from 0 to 1

    for threshold in thresholds:
        # Determine subset of df where confidence is above the threshold
        subset_df = df_sorted[df_sorted['Average Confidence'] >= threshold]

        if not subset_df.empty:
            # Calculate selective accuracy for the current threshold
            correct_predictions = subset_df['Is Correct'].sum()
            total_predictions = len(subset_df)
            selective_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Calculate coverage
            coverage = total_predictions / len(df)
        else:
            selective_accuracy = 1.0  # Assuming perfect accuracy when no predictions are made
            coverage = 0.0

        selective_accuracy_values.append(selective_accuracy)
        coverage_values.append(coverage)

    # Compute AUC
    auc_value = auc(coverage_values, selective_accuracy_values)
    return auc_value

#experiment
def experiment(params):
    import numpy as np
    import os
    import pandas as pd
    from data import MedQA, CommonsenseQA
    import random
    from langchain.prompts import PromptTemplate
    from langchain_community.chat_models import ChatOpenAI

    os.environ.get('OPENAI_API_KEY')
    
    print("Start of Experiment")

    dataset = params["dataset"]
    model = params["model"]
    prompt_template = params["prompt_template"]
    sample_size = params["sample_size"]
    k = params["k"]

    if dataset == "medqa":
        data = MedQA("./datasets/medqa/data/")
        train_data = [x['question'] for x in data._train]
        dev_data = [x['question'] for x in data._dev]
        dev_data_answers = [x['answer'] for x in data._dev]
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
        # dev_examples = dev_data[200:200+sample_size]
        # dev_answers = dev_data_answers[200:200+sample_size]

    if prompt_template == "base":
        template = base_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "atypical":
        template = atypical_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "cot":
        template = cot_prompt_template()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    elif prompt_template == "deliberate_reflection":
        template = base_prompt_template_2()
        prompt = PromptTemplate(template = template, input_variables=['question'])
    else:
        raise ValueError("Invalid prompt template")
    
    if model == "gpt-3.5-turbo":
        gpt = ChatOpenAI(model_name='gpt-3.5-turbo')
        
    llm_chain = LLMChain(
        prompt = prompt,
        llm = gpt
    )

    if prompt_template == "base" or prompt_template == "deliberate_reflection" or prompt_template == "atypical":
        print("Using base prompt template")
        answers_gpt = []
        confidence_scores = []
        ground_truth_probabilities = []
        all_confidence_scores = []
        all_predictions = []
        difficulty_scores = []
        mean_difficulty_scores = []

        i = 0

        for question, correct_answer in tqdm(zip(dev_examples, dev_answers), total=len(dev_examples)):
            temp_scores = []
            temp_answers = []
            temp_difficulty_scores = []

            for _ in range(k):
                raw_answer = llm_chain.run(question)
                standardized_answer, difficulty_score, confidence_score = standardize_and_extract_details(raw_answer)
                if prompt_template == "atypical":
                    atypical_scores = parse_full_output_and_store_scores(raw_answer)
                    print(f"Atypical Scores: {atypical_scores}")
                    if len(atypical_scores) <= 0:
                        atypical_scores = np.array([1])
                    calibrated_confidence = confidence_score * np.mean(np.exp(atypical_scores-1))
                    print(f"Calibrated Confidence: {calibrated_confidence}")
                    confidence_score = calibrated_confidence
    
                temp_answers.append(standardized_answer)
                if difficulty_score is not None:
                    temp_difficulty_scores.append(difficulty_score)
                if confidence_score is not None:
                    temp_scores.append(confidence_score)
                if _ == 0 and i<10:
                    print(f"Question: {question}")
                    print("\n")
                    print(raw_answer)
                    print("\n")
                    print(f"Answer: {standardized_answer}")
                    print(f"Difficulty: {difficulty_score}")
                    print(f"Confidence: {confidence_score}")
                    print("\n")
                    print(f"Correct Answer: {correct_answer}")
                    print("\n")

                    # table = extract_diagnostic_table(raw_answer)
                    # table.to_csv(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}_diagnostics.csv")

            i += 1

            average_difficulty = sum(temp_difficulty_scores) / len(temp_difficulty_scores) if temp_difficulty_scores else None
            average_confidence = sum(temp_scores) / len(temp_scores) if temp_scores else None
            ground_truth_probability = compute_ground_truth_probability(correct_answer, temp_answers)
            majority_answer = get_majority_answer(temp_answers)

            answers_gpt.append(majority_answer)  # Storing only the majority answer
            confidence_scores.append(average_confidence)
            ground_truth_probabilities.append(ground_truth_probability)
            all_confidence_scores.append(temp_scores)
            all_predictions.append(temp_answers)
            difficulty_scores.append(temp_difficulty_scores)
            mean_difficulty_scores.append(average_difficulty)

        # Creating DataFrame
        df = pd.DataFrame({
            'Questions': dev_examples,
            'Difficulty Score': difficulty_scores,
            'Mean Difficulty Score': mean_difficulty_scores, # 'Mean Difficulty Score' is the average of the difficulty scores for each question
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
    print("Computing accuracy")
    correct = 0
    for i in range(len(df)):
        if df['Correct Answers'][i] == df['Majority Predicted Answer'][i]:
            correct += 1
    accuracy = correct / len(df)
    acc = np.array([accuracy])
    print(accuracy)
    np.save(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}_acc.npy", acc)

    #compute ece
    print("Computing ece")
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
    print(ece)
    np.save(f"./results/{dataset}_{model}_{prompt_template}_{sample_size}_{current_time}_ece.npy", ece)

    print("Computing AUC")
    auc_score = compute_auc(df)
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