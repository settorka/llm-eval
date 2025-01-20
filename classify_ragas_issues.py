# Refined Model (Threshold Evaluator + Probability Estimator)
def evaluate_thresholds_and_probabilities(row):
    noise_sensitivity, context_recall, context_precision, context_entities_recall, answer_relevancy, faithfulness = row
    
    # Probabilities for each issue type
    retrieval_prob = 0
    context_prob = 0
    generation_prob = 0
    
    # Rule for Retrieval (Noise Sensitivity)
    if noise_sensitivity < 0.5:
        retrieval_prob = 0.2
    elif noise_sensitivity >= 0.5:
        retrieval_prob = 0.6
    
    # Rule for Generation (Answer Relevancy and Faithfulness)
    if answer_relevancy < 0.6 or faithfulness < 0.6:
        generation_prob = 0.5
    elif answer_relevancy >= 0.6 and faithfulness >= 0.6:
        generation_prob = 0.3
    
    # Rule for Context (Entities Recall and Context Recall)
    if context_entities_recall < 0.6:
        context_prob = 0.5
    elif context_entities_recall >= 0.6:
        context_prob = 0.4

    # Final adjustments (normalizing the sum of probabilities to 1)
    total = retrieval_prob + context_prob + generation_prob
    retrieval_prob /= total
    context_prob /= total
    generation_prob /= total

    return {
        'retrieval_prob': retrieval_prob,
        'context_prob': context_prob,
        'generation_prob': generation_prob
    }

def process_rows(rows):
    result = []
    for row in rows:
        result.append(evaluate_thresholds_and_probabilities(row))
    return result

# New Sample Input Data (RAGAS scores for 2 new cases)
new_data = [
    [0.7, 0.5, 0.6, 0.8, 0.5, 0.6],  # RAGAS_ID 1
    [0.3, 0.9, 0.8, 0.6, 0.8, 0.9]   # RAGAS_ID 2
]

# Run the process with the new data
new_results = process_rows(new_data)

# Output the results
for idx, res in enumerate(new_results, start=1):
    print(f"RAGAS_ID {idx}: Retrieval: {res['retrieval_prob']:.2f}, Context: {res['context_prob']:.2f}, Generation: {res['generation_prob']:.2f}")
