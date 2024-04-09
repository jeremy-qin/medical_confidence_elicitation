from datasets import load_dataset
import json
import string


class MedQA():
  def __init__(self, json_path, **kwargs):
    self._dev_data = load_dataset('json', data_files=json_path+"dev.json", field='data')['train']
    self._train_data = load_dataset('json', data_files=json_path+"train.json", field='data')['train']
    self._test_data = load_dataset('json', data_files=json_path+"test.json", field='data')['train']
    super().__init__(**kwargs)

  @property
  def _dev(self):
    return self._dev_data

  @property
  def _train(self):
    return self._train_data

  @property
  def _test(self):
    return self._test_data

  def __len__(self):
    return len(self.data)
  
class CommonsenseQA():
  def __init__(self, json_path, **kwargs):
    self._dev_data, self._dev_answers = self.load_commonsense_qa(json_path)
    # self._train_data = load_dataset('json', data_files=json_path+"train.json", field='data')['train']
    # self._test_data = load_dataset('json', data_files=json_path+"test.json", field='data')['train']
    super().__init__(**kwargs)

  @property
  def _dev_questions(self):
    return self._dev_data

  @property
  def _dev_labels(self):
    return self._dev_answers
  
  # @property
  # def _train(self):
  #   return self._train_data

  # @property
  # def _test(self):
  #   return self._test_data

  def __len__(self):
    return len(self.data)
  
  def load_commonsense_qa(self, path):
    questions_with_choices = []

    # List to store the correct answers
    correct_answers = []

    # Open the file and read line by line
    with open(path) as h:
        for line in h:
            example = json.loads(line)  # Parse the JSON object from each line

            # Construct the question text with its choices
            question_text = example['question']['stem']
            choices_text = [f"{string.ascii_uppercase[i]}: {choice['text']}" for i, choice in enumerate(example['question']['choices'])]
            question_and_choices = question_text + ' ' + ' '.join(choices_text)
            
            # Add the constructed question and its choices to the list
            questions_with_choices.append(question_and_choices)

            # The correct answer key (e.g., "A", "B", "C", etc.) is directly added to the correct_answers list
            correct_answers.append(example['answerKey'])
    return questions_with_choices, correct_answers