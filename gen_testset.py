import os
from langchain_community.document_loaders import WebBaseLoader
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Set your Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-key"

azure_configs = {
    "base_url": "your-azure-endpoint",  # e.g., "https://your-resource-name.openai.azure.com"
    "model_deployment": "your-model-deployment",  # e.g., "gpt-4-deployment"
    "model_name": "gpt-4",  # e.g., "gpt-4"
    "embedding_deployment": "your-embedding-deployment",  # e.g., "text-embedding-ada-002"
    "embedding_name": "text-embedding-ada-002",  # e.g., "text-embedding-ada-002"
}

# Initialize the LLM and embedding wrappers
llm = LangchainLLMWrapper(
    AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )
)

embedding_model = LangchainEmbeddingsWrapper(
    AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )
)

# Load the documents from an online PDF link
pdf_url = "https://your-pdf-link.com/document.pdf"
loader = WebBaseLoader(pdf_url)

documents = loader.load()

# Generate the testset
generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
testset_size = 10  # Number of queries to generate for the testset
testset = generator.generate_with_langchain_docs(documents, testset_size=testset_size)

# Export the testset to a pandas DataFrame for analysis
import pandas as pd

testset_df = testset.to_pandas()

# Save the DataFrame to a CSV file
testset_df.to_csv("testset.csv", index=False)

print("Testset generated and saved to testset.csv")
