from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall, BleuScore, RougeScore
from ragas.dataset_schema import SingleTurnSample
from langchain.chat_models import AzureChatOpenAI


# Abstract Base Class for Evaluators
class LLMEvaluator(ABC):
    @abstractmethod
    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to evaluate LLM's performance.
        """
        pass


# Chained Evaluator to handle multiple evaluators
class ChainedLLMEvaluator(LLMEvaluator):
    def __init__(self, evaluators: List[LLMEvaluator]):
        """
        Initializes the evaluator with a list of evaluators to chain.
        """
        self.evaluators = evaluators

    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the dataset using all evaluators in the chain.
        The result is concatenated into a single DataFrame.
        """
        all_results = []
        for evaluator in self.evaluators:
            result = evaluator.evaluate(dataset)
            all_results.append(result)

        # Concatenate all results into a single DataFrame
        final_result = pd.concat(all_results, axis=1)
        return final_result


# Builder class for constructing a chain of evaluators
class LLMEvaluatorBuilder:
    def __init__(self):
        """
        Initializes an empty list of evaluators.
        """
        self.evaluators = []

    def add_evaluator(self, evaluator: LLMEvaluator) -> 'LLMEvaluatorBuilder':
        """
        Adds an evaluator to the evaluation pipeline.
        """
        self.evaluators.append(evaluator)
        return self

    def build(self) -> ChainedLLMEvaluator:
        """
        Builds and returns the chained evaluator with the added evaluators.
        """
        return ChainedLLMEvaluator(self.evaluators)


# Concrete Evaluator using RAGAS
class RagasLLMEvaluator(LLMEvaluator):
    def __init__(self, metrics: List = None, azure_endpoint=None, azure_api_key=None, azure_model_name=None):
        """
        Initializes RAGAS evaluator with specific metrics and AzureOpenAI client.
        """
        if metrics is None:
            metrics = [faithfulness, answer_correctness, context_precision, context_recall]
        self.metrics = metrics

        if azure_endpoint and azure_api_key and azure_model_name:
            # Initialize AzureOpenAI client
            self.llm = AzureChatOpenAI(
                azure_openai_api_key=azure_api_key,
                azure_openai_api_base=azure_endpoint,
                azure_openai_api_version="2024-02-01",
                deployment_name=azure_model_name,
                temperature=0
            )
        else:
            raise ValueError("Azure endpoint, API key, and model name must be provided.")

    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the LLM based on the specified metrics using AzureOpenAI.
        """
        # Pass the AzureOpenAI client to RAGAS
        score = evaluate(dataset, metrics=self.metrics, llm=self.llm)
        return score.to_pandas()


# Concrete Evaluator using BLEU Score
class BleuScoreEvaluator(LLMEvaluator):
    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the LLM using BLEU score metric.
        """
        bleu_scorer = BleuScore()
        bleu_scores = []

        for _, sample in dataset.iterrows():
            response = sample['answer']
            reference = sample['ground_truth']
            single_turn = SingleTurnSample(response=response, reference=reference)
            bleu_score = bleu_scorer.single_turn_ascore(single_turn)
            bleu_scores.append(bleu_score)

        return pd.DataFrame({'BLEU_Score': bleu_scores})


# Concrete Evaluator using ROUGE Score
class RougeScoreEvaluator(LLMEvaluator):
    def evaluate(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the LLM using ROUGE score metric.
        """
        rouge_scorer = RougeScore()
        rouge_scores = []

        for _, sample in dataset.iterrows():
            response = sample['answer']
            reference = sample['ground_truth']
            single_turn = SingleTurnSample(response=response, reference=reference)
            rouge_score = rouge_scorer.single_turn_ascore(single_turn)
            rouge_scores.append(rouge_score)

        return pd.DataFrame({'ROUGE_Score': rouge_scores})
