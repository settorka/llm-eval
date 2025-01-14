import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List

# Abstract Base Class for LLM Results Assessment
class LLMResultsAssessment(ABC):
    @abstractmethod
    def assess(self, dataset: pd.DataFrame, standard: Dict[str, float]) -> pd.DataFrame:
        """
        Abstract method to assess the results in the dataset based on the standard.
        """
        pass

# Concrete Implementation using RAGAS metrics
class RagasLLMResultsAssessment(LLMResultsAssessment):
    def __init__(self):
        """
        Initializes the RagasLLMResultsAssessment with default RAGAS metrics.
        """
        # Metrics mapping could be extended or passed as parameters
        self.metrics = ['faithfulness', 'answer_correctness', 'context_precision', 'context_recall']
    
    def assess(self, dataset: pd.DataFrame, standard: Dict[str, float]) -> pd.DataFrame:
        """
        Assesses the LLM evaluation results against a threshold standard.
        """
        analysis_results = []

        # Iterate through each metric in the dataset and compare it with the standard
        for metric in self.metrics:
            threshold = standard.get(metric, 0.8)  # Default to 0.8 if not specified in the standard
            result = []

            # Loop through the scores for this metric and apply the threshold
            for score in dataset[metric]:
                if score >= threshold:
                    result.append("Good")
                else:
                    result.append("Not Good")
            
            # Add the results to the dataframe
            analysis_results.append(pd.Series(result, name=f"{metric}_grade"))

        # Combine the original dataset with the analysis results
        analysis_df = pd.concat([dataset, *analysis_results], axis=1)
        return analysis_df
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List

# Abstract Base Class for LLM Results Assessment
class LLMResultsAssessment(ABC):
    @abstractmethod
    def assess(self, dataset: pd.DataFrame, standard: Dict[str, float]) -> pd.DataFrame:
        """
        Abstract method to assess the results in the dataset based on the standard.
        """
        pass

# Concrete Implementation using RAGAS metrics
class RagasLLMResultsAssessment(LLMResultsAssessment):
    def __init__(self):
        """
        Initializes the RagasLLMResultsAssessment with default RAGAS metrics.
        """
        # Metrics mapping could be extended or passed as parameters
        self.metrics = ['faithfulness', 'answer_correctness', 'context_precision', 'context_recall']
    
    def assess(self, dataset: pd.DataFrame, standard: Dict[str, float]) -> pd.DataFrame:
        """
        Assesses the LLM evaluation results against a threshold standard.
        """
        analysis_results = []

        # Iterate through each metric in the dataset and compare it with the standard
        for metric in self.metrics:
            threshold = standard.get(metric, 0.8)  # Default to 0.8 if not specified in the standard
            result = []

            # Loop through the scores for this metric and apply the threshold
            for score in dataset[metric]:
                if score >= threshold:
                    result.append("Good")
                else:
                    result.append("Not Good")
            
            # Add the results to the dataframe
            analysis_results.append(pd.Series(result, name=f"{metric}_grade"))

        # Combine the original dataset with the analysis results
        analysis_df = pd.concat([dataset, *analysis_results], axis=1)
        return analysis_df
