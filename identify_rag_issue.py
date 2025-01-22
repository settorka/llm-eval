def evaluate_thresholds_and_probabilities(row):
    context_recall, context_precision, context_entities_recall, answer_relevancy, faithfulness = row

    # Adjusted weights based on the provided rationale
    retrieval_weight = context_entities_recall  # Retrieval is tied to entities recall
    context_weight = (1 - context_entities_recall) * (context_recall + context_precision) / 2
    generation_weight = (1 - ((answer_relevancy + faithfulness) / 2))  # Penalize low relevancy/faithfulness

    # Boost context weight when context_entities_recall is very low (if context entities recall < 0.5)
    if context_entities_recall < 0.5:
        context_weight *= 1.5

    # Boost generation weight for very low answer relevancy or faithfulness
    if min(answer_relevancy, faithfulness) < 0.4:
        generation_weight *= 2  # Increased the multiplier here to boost generation even more

    # Adjust retrieval weight when generation is dominant (when generation_weight is much higher)
    if generation_weight > retrieval_weight * 1.5:
        retrieval_weight *= 0.6  # Lower retrieval weight further if generation is dominant

    # Further boost context if both retrieval and generation are problematic
    if retrieval_weight < 0.1 and generation_weight < 0.1:
        context_weight *= 2  # If retrieval and generation both fail, heavily weight context
    
    # Normalize the weights to sum to 1
    total_weight = retrieval_weight + context_weight + generation_weight
    retrieval_prob = retrieval_weight / total_weight
    context_prob = context_weight / total_weight
    generation_prob = generation_weight / total_weight

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

# New Sample Input Data with RAGAS_ID 4 included
new_data = [
    [0.5, 0.6, 0.8, 0.5, 0.6],  # RAGAS_ID 1
    [0.9, 0.8, 0.2, 0.8, 0.9],  # RAGAS_ID 2
    [0.9, 0.8, 0.8, 0.0, 0.0],  # RAGAS_ID 3 

]

# Run the process with the new data
new_results = process_rows(new_data)

# Output the results
for idx, res in enumerate(new_results, start=1):
    print(f"RAGAS_ID {idx}: Retrieval: {res['retrieval_prob']:.2f}, Context: {res['context_prob']:.2f}, Generation: {res['generation_prob']:.2f}")
