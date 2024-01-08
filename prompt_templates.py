# Prompt template for confidence elicitation

def base_prompt_template():
    template = """
        Question and Options: {question}
        Answer with the letter of the choice followed by a colon and the answer text, then provide a confidence score in percentage between 0 and 100.

        Answer and Confidence:
        """
    return template

def cot_prompt_template():
    template = """

        Question and Options: {question}

        Let's think step by step.
        [Outline the reasoning process here, going through the question step by step.]

        Based on the reasoning above, complete the following:
        The answer is letter __ and I am __% confident.

        Reasoning, Answer and Confidence:
        """
    return template