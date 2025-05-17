# Data Generation for GSM8k

## Files

- `generate_questions.py`: Extract equations, permute numbers, and generate questions using QwQ-32B.
- `filter_questions.py`: Discard generated questions which are naive, not finished or do not follow the format of the origin data.
- `generate_answers.py`: Generate answers for filtered questions using Qwen2.5-7B, following the CoT format.
- `filter_answers.py`: Discard answers which do not match the ground truth.