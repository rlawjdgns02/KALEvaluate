# mmlu/test.py - 10개 문제 테스트

from datasets import load_from_disk
import ollama
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mmlu')

# 데이터 로드
dataset = load_from_disk(DATA_PATH)

cnt = 0

# 10개 테스트
for i in range(10):
    example = dataset['test'][i]

    prompt = f"""The following is a multiple choice question about {example['subject'].replace('_', ' ')}.

Question: {example['question']}
(A) {example['choices'][0]}
(B) {example['choices'][1]}
(C) {example['choices'][2]}
(D) {example['choices'][3]}

Answer with just the letter (A, B, C, or D):"""

    response1 = ollama.generate(
        model='llama3.2:latest',
        prompt=prompt,
        options={'temperature': 0}
    )

    answer = response1['response'].strip()
    correct = chr(65 + example['answer'])

    print(f"[{i+1}] 정답: {correct} | 응답: {answer[:50]}{'...' if len(answer) > 50 else ''}")


print(dataset['test'][0]['subject'])