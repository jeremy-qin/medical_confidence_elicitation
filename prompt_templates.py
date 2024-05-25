# Prompt template for confidence elicitation

# def base_prompt_template():
#     template = """
#         Question and Options: {question}
#         Answer with the letter of the choice followed by a colon and the answer text, then provide a confidence score in percentage between 0 and 100.

#         Answer and Confidence:
#         """
#     return template

def base_prompt_template():
    template = """
        Question and Options: {question}
        Please provide your response in the following format:
        - Answer (letter): [Letter of the choice]
        - Difficulty: [Score on a scale from 1 to 10 with 10 being the hardest]
        - Confidence: [Percentage score between 0 and 100%]

        Answer, Difficulty and Confidence:
        """
    return template

def atypical_prompt_template():
    template = """
        Question and Options: {question}

        First, assess each symptom with respect to its typicality in the described scenario. Assign an atypicality score between 0 and 1, where:
        - 0 indicates a highly atypical symptom, uncommon or rare in such scenarios.
        - 1 indicates a very typical symptom, commonly expected in such scenarios.
        - Scores between 0 and 1 (such as 0.25, 0.5, 0.75) indicate varying degrees of typicality.

        Symptoms and signs:
        - Symptom 1: [Atypical score]
        - Symptom 2: [Atypical score]
        - Symptom 3: [Atypical score]
        - ...

        Then, provide your response in the following format:
        Response:
        - Answer (letter): [Letter of the choice]
        - Difficulty: [Score on a scale from 1 to 10 with 10 being the hardest]
        - Confidence: [Percentage score between 0 and 100%]

        Answer, Difficulty and Confidence:
        """
    return template

def atypical_situation_prompt_template():
    template = """
        Question and Options: {question}

        First, assess the situation described in the question and assign an atypicality score between 0 and 1, where:
        - 0 indicates a highly atypical situation, uncommon or rare in such scenarios.
        - 1 indicates a very typical situation, commonly expected in such scenarios.
        - Scores between 0 and 1 (such as 0.25, 0.5, 0.75) indicate varying degrees of typicality.

        Situation Atypicality: [Atypicality score]

        Then, provide your response in the following format:
        Response:
        - Answer (letter): [Letter of the choice]
        - Difficulty: [Score on a scale from 1 to 10 with 10 being the hardest]
        - Confidence: [Percentage score between 0 and 100%]

        Answer, Difficulty, and Confidence:
        """
    return template

def commonsense_qa_prompt_template():
    template = """
        Question and Options: {question}

        First, list all relevant commonsense elements related to the question and assign each an atypicality score between 0 and 1 (0 being highly atypical or unexpected in a typical scenario, and 1 being very typical or expected):

        Commonsense Elements:
        - Element 1: [Atypicality score]
        - Element 2: [Atypicality score]
        - Element 3: [Atypicality score]
        - ...

        Consider factors such as actions, consequences, social norms, physical interactions, and logical sequences when assessing atypicality.

        Then, provide your response in the following format:
        Response:
        - Answer (letter): [Letter of the choice]
        - Reasoning Atypicality: [Average Atypicality score of used elements]
        - Difficulty: [Score on a scale from 1 to 10, with 10 being the hardest based on the complexity of commonsense reasoning required]
        - Confidence: [Adjusted confidence percentage between 0 and 100%, considering the Reasoning Atypicality and question difficulty]

        Answer, Reasoning Atypicality, Difficulty, and Confidence:
        """
    return template

def base_prompt_template_2():
    template = """
        Question and Options: {question}

        Start with an initial diagnostic hypothesis with its rationale and state it as follow:
        - Initial diagnostic hypothesis: [First hypothesis]
        - Rationale: [Reasoning for the initial hypothesis]

        Now consider alternative hypotheses. For each diagnostic hypothesis including the initial hypothesis, list the supporting and contradictory clinical findings, mention any missing findings that are expected, and rate the likelihood.
        When listing the findings that speak in favor of or against each hypothesis, rate the importance of each findings as high, medium, or low.
        
        Based on this, please generate a diagnostic table with the following format:
        
        - Diagnostic hypotheses:
            1. [First hypothesis]
            2. [Second hypothesis]
            3. [Third hypothesis]
            ... [Additional hypotheses as necessary]

        - Findings that speak in favor of each hypothesis:
            1. [First hypothesis]: [Supporting findings]
            2. [Second hypothesis]: [Supporting findings]
            3. [Third hypothesis]: [Supporting findings]
            ... [Corresponding findings for additional hypotheses]

        - Findings that speak against each hypothesis:
            1. [First hypothesis]: [Contradictory findings]
            2. [Second hypothesis]: [Contradictory findings]
            3. [Third hypothesis]: [Contradictory findings]
            ... [Corresponding findings for additional hypotheses]

        - Findings expected to be present but not described in the case:
            1. [First hypothesis]: [Expected but missing findings]
            2. [Second hypothesis]: [Expected but missing findings]
            3. [Third hypothesis]: [Expected but missing findings]
            ... [Corresponding expected findings for additional hypotheses]

        - Likelihood:
            1. [First hypothesis]: [Likelihood assessment]
            2. [Second hypothesis]: [Likelihood assessment]
            3. [Third hypothesis]: [Likelihood assessment]
            ... [Corresponding likelihood assessments for additional hypotheses]

        Then answer the question by providing your response in the following format:
        - Answer (letter): [Letter of the choice]
        - Difficulty: [Score on a scale from 1 to 10 with 10 being the hardest]
        - Confidence: [Percentage score between 0 and 100%]

        Here are some examples. Use these to analogically compare the difficulty and confidence of your question:

        Example 1:
        Question and Options: A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given? A:Chloramphenicol, B:Gentamicin, C:Ciprofloxacin, D:Ceftriaxone, E:Trimethoprim
          - Answer (letter): D
          - Difficulty: 5
          - Confidence: 90%

        Answer, Difficulty and Confidence:
        """
    return template

# def cot_prompt_template():
#     template = """

#         Question and Options: {question}

#         Let's think step by step.
#         [Outline the reasoning process here, going through the question step by step.]

#         Based on the reasoning above, complete the following:
#         The answer is letter __ and I am __% confident.

#         Reasoning, Answer and Confidence:
#         """
#     return template

def cot_prompt_template():
    template = """
        Question and Options: {question}

        Let's think step by step.
#       [Outline the reasoning process here, going through the question step by step.]

        Then, provide your response in the following format:
        Response:
        - Answer (letter): [Letter of the choice]
        - Difficulty: [Score on a scale from 1 to 10 with 10 being the hardest]
        - Confidence: [Percentage score between 0 and 100%]

        Answer, Difficulty and Confidence:
        """
    return template