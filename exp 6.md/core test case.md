def check_pos_tags(test_sentences, predicted_tags, true_tags):
    results = []
    for i, sentence in enumerate(test_sentences):
        pred_tags = predicted_tags[i]
        true_tags_for_sentence = true_tags[i]
        correct = "Y" if pred_tags == true_tags_for_sentence else "N"
        results.append((sentence, pred_tags, true_tags_for_sentence, correct))

    return results
test_sentences = [
    "I love NLP",
    "He plays football"
]
predicted_tags = [
    ['PRON', 'VERB', 'NOUN'],
    ['PRON', 'VERB', 'NOUN']
] # Added comma here
true_tags = [
    ['PRON', 'VERB', 'NOUN'],  # Added comma here
    ['PRON', 'VERB', 'NOUN']
]
results = check_pos_tags(test_sentences, predicted_tags, true_tags)
for sentence, pred_tags, true_tags_for_sentence, correct in results:
    print(f"Sentence: {sentence}")
    print(f"Predicted Tags: {pred_tags}")
    print(f"True Tags: {true_tags_for_sentence}")
    print(f"Correct (Y/N): {correct}")
    print("-" * 50)
    output<img width="370" height="177" alt="Screenshot 2025-09-24 111137" src="https://github.com/user-attachments/assets/6b407e12-814c-4e1a-be1d-98d549c6f34c" />
