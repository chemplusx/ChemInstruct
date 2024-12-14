from unittest import TestCase

from nerllama.utils.metric import calculate_metrics, extract_classes

class TestMetrics(TestCase):

  def test_calculate_metrics(self):

    extracted_set = [
      {"chemical": "calcium"},
      {"chemical": "Vitamin B12"},
      {"chemical": "1,2-chloromethylcyclohexane"},
      {"chemical": "Nitrogen"},
      {"chemical": "magnesium phosphate"},
      {"chemical": "water"},
      {"chemical": "O2"}
    ]

    target_entities = [
      {"chemical": "Ca"},
      {"chemical": "calcium"},
      {"chemical": "Vitamin B12"},
      {"chemical": "1,2-chloromethylcyclohexane"},
      {"chemical": "Nitrogen"},
      {"chemical": "magnesium phosphate"},
      {"chemical": "Mg"},
      {"chemical": "water"},
      {"chemical": "O2"}
    ]

    result = calculate_metrics(
      extracted_entities=extracted_set,
      target_entities=target_entities,
      entity_types=['chemical'],
      split_entities=False
    )

    expected_output = {
      'chemical': {
        'precision': 0.4, 
        'recall': 0.42105263157894735, 
        'f1': 0.41025641025641024
      }, 
      'overall': {
        'precision': 0.4, 
        'recall': 0.42105263157894735, 
        'f1': 0.41025641025641024
      }, 
      'loss': 0.5897435897435898
    }

    self.assertEqual(result, expected_output)

  
  def test_extract_classes(self):
    generated_string = """### Task: You are solving the Chemical Named Entity Recognition problem. Extract from the text words related to chemical entities 
    ### Input: Extract all chemical entities from the given text. Text: Iron is required for proper functioning of vital organs and oxygen as well.
    ### Answer:  Iron: chemical\noxygen: chemical\n \n
    """

    result = extract_classes(generated_string, entity_types=['chemical'])

    self.assertEqual(result['chemical'], ['Iron: chemical'])
