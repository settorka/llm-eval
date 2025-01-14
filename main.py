# main.py (Example Usage)

from datasets import Dataset
import pandas as pd
from model_eval import LLMEvaluatorBuilder, RagasLLMEvaluator, BleuScoreEvaluator, RougeScoreEvaluator

# Sample data for evaluation
data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts': [
        ['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967.'],
        ['The Green Bay Packers have won a record number of super bowls.']
    ],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
}

dataset = Dataset.from_dict(data_samples)
df = dataset.to_pandas()

# Use the builder pattern to construct a chain of evaluators
builder = LLMEvaluatorBuilder()
chained_evaluator = (builder
                     .add_evaluator(RagasLLMEvaluator())  # Add RAGAS evaluator
                     .add_evaluator(BleuScoreEvaluator())  # Add BLEU score evaluator
                     .add_evaluator(RougeScoreEvaluator())  # Add ROUGE score evaluator
                     .build())  # Build the final chained evaluator

# Evaluate the dataset
result = chained_evaluator.evaluate(df)

# Output the final result as a DataFrame
print(result)
