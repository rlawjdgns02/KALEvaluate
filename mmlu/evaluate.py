# mmlu/evaluate.py

from datasets import load_from_disk
import ollama
import re
from tqdm import tqdm
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mmlu')


def extract_answer(response_text):
    """응답에서 A/B/C/D 추출 (마지막으로 나오는 것)"""
    matches = re.findall(r'\b([A-D])\b', response_text.upper())
    if matches:
        return ord(matches[-1]) - ord('A')  # 마지막 매칭 사용
    return -1


def create_prompt(subject, question, choices):
    """프롬프트 생성"""
    return f"""The following is a multiple choice question about {subject.replace('_', ' ')}.

Question: {question}
(A) {choices[0]}
(B) {choices[1]}
(C) {choices[2]}
(D) {choices[3]}

Answer with just the letter (A, B, C, or D):"""


def get_model_answer(prompt, model_name):
    """모델 호출"""
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={'temperature': 0}
        )
        return response['response']
    except Exception as e:
        print(f"Error: {e}")
        return ""


def evaluate(model_name='qwen3:4b', subjects=None, limit=None):
    """MMLU 평가 실행"""

    dataset = load_from_disk(DATA_PATH)
    test_data = dataset['test']

    # 주제 필터링
    if subjects:
        test_data = test_data.filter(lambda x: x['subject'] in subjects)

    # 개수 제한
    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))

    correct = 0
    total = 0
    failed = 0

    for item in tqdm(test_data, desc="Evaluating", ncols=80):
        prompt = create_prompt(item['subject'], item['question'], item['choices'])
        response = get_model_answer(prompt, model_name)
        predicted = extract_answer(response)

        if predicted == -1:
            failed += 1
        elif predicted == item['answer']:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Failed extractions: {failed}")
    print(f"{'='*50}")

    return {
        'model': model_name,
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'failed': failed
    }


if __name__ == "__main__":
    # 전체 평가
    evaluate(model_name='qwen3:4b')
