from datasets import load_dataset

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