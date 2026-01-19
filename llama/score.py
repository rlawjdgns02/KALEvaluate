import os
import re
import io
import base64
import pandas as pd
from PIL import Image
from collections import Counter
from torchvision import datasets, transforms
import glob
import ollama

# 1. 모델 설정 (ollama pull llama3.2-vision:11b 먼저 실행 필요)
MODEL_NAME = "llama3.2-vision:11b"

# 2. MNIST 데이터 로드 (프로젝트 공통 경로 사용)
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 3. Prompt Robustness용 10가지 프롬프트
PROMPTS = [
    "What digit is shown in this image? Answer with only a single number (0-9).",
    "Identify the handwritten digit in this image. Reply with just one number from 0 to 9.",
    "Look at this image and tell me which digit it shows. Respond with a single digit only.",
    "What number do you see in this picture? Give me only the digit (0-9).",
    "Please recognize the digit displayed in this image. Answer with one number only.",
    "This image contains a handwritten digit. What is it? Reply with just 0-9.",
    "Can you identify the number shown here? Respond with a single digit.",
    "What is the digit written in this image? Only output the number (0-9).",
    "Examine this image and determine the digit. Answer with exactly one number.",
    "Tell me which digit (0-9) is depicted in this image. Single number response only.",
]

# 4. Choice Order Robustness용 선택지 순서 변형 (5가지)
CHOICE_ORDERS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 원래 순서
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],  # 역순
    [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],  # 짝수 먼저
    [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],  # 5부터 시작
    [3, 1, 4, 1, 5, 9, 2, 6, 8, 0],  # 무작위 (중복 제거: 3,1,4,5,9,2,6,8,0,7)
]
# 중복 제거된 무작위 순서로 수정
CHOICE_ORDERS[4] = [3, 1, 4, 5, 9, 2, 6, 8, 0, 7]


def extract_digit(text):
    """응답에서 숫자 추출"""
    match = re.search(r'\d', text)
    return int(match.group()) if match else None


def image_to_base64(image_tensor):
    """텐서 이미지를 base64로 변환"""
    image_np = (image_tensor.squeeze().numpy() * 255).astype('uint8')
    pil_image = Image.fromarray(image_np, mode='L').convert('RGB')

    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def query_model(image_b64, prompt, temperature=0.7):
    """Ollama 모델에 쿼리"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        options={
            "temperature": temperature,
        }
    )
    return response['message']['content']


def calculate_prompt_robustness(image_tensor, n_prompts=10):
    """
    Prompt Robustness: 10가지 다른 프롬프트로 테스트

    Args:
        image_tensor: 토치 텐서 이미지
        n_prompts: 사용할 프롬프트 수 (최대 10)

    Returns:
        score: 일관성 점수 (0~1)
        most_common_pred: 가장 많이 나온 예측
        predictions: 모든 예측 리스트
    """
    predictions = []
    image_b64 = image_to_base64(image_tensor)

    for i in range(min(n_prompts, len(PROMPTS))):
        text = query_model(image_b64, PROMPTS[i], temperature=0.0)  # greedy decoding
        digit = extract_digit(text)
        predictions.append(digit)

    valid_preds = [p for p in predictions if p is not None]
    if valid_preds:
        counter = Counter(valid_preds)
        most_common_pred, count = counter.most_common(1)[0]
        score = count / len(predictions)
        return score, most_common_pred, predictions
    return 0, None, predictions


def calculate_nongreedy_robustness(image_tensor, n_samples=5):
    """
    Non-Greedy Robustness: 동일 프롬프트 + temperature=0.7로 N번 샘플링

    Args:
        image_tensor: 토치 텐서 이미지
        n_samples: 샘플링 횟수

    Returns:
        score: 일관성 점수 (0~1)
        most_common_pred: 가장 많이 나온 예측
        predictions: 모든 예측 리스트
    """
    predictions = []
    image_b64 = image_to_base64(image_tensor)
    prompt = PROMPTS[0]  # 고정 프롬프트

    for _ in range(n_samples):
        text = query_model(image_b64, prompt, temperature=0.7)
        digit = extract_digit(text)
        predictions.append(digit)

    valid_preds = [p for p in predictions if p is not None]
    if valid_preds:
        counter = Counter(valid_preds)
        most_common_pred, count = counter.most_common(1)[0]
        score = count / n_samples
        return score, most_common_pred, predictions
    return 0, None, predictions


def calculate_choice_order_robustness(image_tensor, n_orders=5):
    """
    Choice Order Robustness: 선택지 순서를 바꿔서 테스트
    MNIST는 MCQ가 아니므로, 프롬프트에 선택지 순서를 명시하여 테스트

    Args:
        image_tensor: 토치 텐서 이미지
        n_orders: 테스트할 순서 변형 수 (최대 5)

    Returns:
        score: 일관성 점수 (0~1)
        most_common_pred: 가장 많이 나온 예측
        predictions: 모든 예측 리스트
    """
    predictions = []
    image_b64 = image_to_base64(image_tensor)

    for i in range(min(n_orders, len(CHOICE_ORDERS))):
        order = CHOICE_ORDERS[i]
        choices_str = ", ".join(map(str, order))
        prompt = f"What digit is shown in this image? Choose from these options in order: [{choices_str}]. Answer with only the digit."

        text = query_model(image_b64, prompt, temperature=0.0)  # greedy decoding
        digit = extract_digit(text)
        predictions.append(digit)

    valid_preds = [p for p in predictions if p is not None]
    if valid_preds:
        counter = Counter(valid_preds)
        most_common_pred, count = counter.most_common(1)[0]
        score = count / len(predictions)
        return score, most_common_pred, predictions
    return 0, None, predictions


def evaluate(n_samples_eval=100):
    """
    SCORE 방식으로 MNIST 평가 (3가지 robustness 모두 적용)

    Args:
        n_samples_eval: 평가할 테스트 샘플 수
    """
    results = []

    for idx in range(n_samples_eval):
        image, true_label = test_dataset[idx]

        # 1. Prompt Robustness
        pr_score, pr_pred, pr_preds = calculate_prompt_robustness(image, n_prompts=10)

        # 2. Non-Greedy Robustness
        ng_score, ng_pred, ng_preds = calculate_nongreedy_robustness(image, n_samples=5)

        # 3. Choice Order Robustness
        co_score, co_pred, co_preds = calculate_choice_order_robustness(image, n_orders=5)

        results.append({
            'idx': idx,
            'true_label': true_label,
            # Prompt Robustness
            'pr_score': pr_score,
            'pr_prediction': pr_pred,
            'pr_correct': pr_pred == true_label,
            'pr_all_predictions': pr_preds,
            # Non-Greedy Robustness
            'ng_score': ng_score,
            'ng_prediction': ng_pred,
            'ng_correct': ng_pred == true_label,
            'ng_all_predictions': ng_preds,
            # Choice Order Robustness
            'co_score': co_score,
            'co_prediction': co_pred,
            'co_correct': co_pred == true_label,
            'co_all_predictions': co_preds,
        })

        print(f"Sample {idx}: True={true_label} | "
              f"PR: pred={pr_pred}, score={pr_score:.2f}, correct={pr_pred == true_label} | "
              f"NG: pred={ng_pred}, score={ng_score:.2f}, correct={ng_pred == true_label} | "
              f"CO: pred={co_pred}, score={co_score:.2f}, correct={co_pred == true_label}")

    return results


def analyze_results(results):
    """결과 분석 및 출력"""
    df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("SCORE 평가 결과 (MNIST + LLaMA Vision)")
    print("="*70)

    # 각 Robustness별 개별 통계
    print("\n=== Prompt Robustness (PR) ===")
    print(f"Accuracy: {df['pr_correct'].mean():.3f}")
    print(f"Consistency Score: {df['pr_score'].mean():.3f}")
    print(f"Score 분포: min={df['pr_score'].min():.2f}, max={df['pr_score'].max():.2f}, std={df['pr_score'].std():.3f}")

    print("\n=== Non-Greedy Robustness (NG) ===")
    print(f"Accuracy: {df['ng_correct'].mean():.3f}")
    print(f"Consistency Score: {df['ng_score'].mean():.3f}")
    print(f"Score 분포: min={df['ng_score'].min():.2f}, max={df['ng_score'].max():.2f}, std={df['ng_score'].std():.3f}")

    print("\n=== Choice Order Robustness (CO) ===")
    print(f"Accuracy: {df['co_correct'].mean():.3f}")
    print(f"Consistency Score: {df['co_score'].mean():.3f}")
    print(f"Score 분포: min={df['co_score'].min():.2f}, max={df['co_score'].max():.2f}, std={df['co_score'].std():.3f}")

    # SCORE와 정확도의 관계 (각 방식별)
    print("\n=== SCORE와 정확도 관계 ===")
    for name, score_col, correct_col in [('PR', 'pr_score', 'pr_correct'),
                                          ('NG', 'ng_score', 'ng_correct'),
                                          ('CO', 'co_score', 'co_correct')]:
        high = df[df[score_col] >= 0.8]
        low = df[df[score_col] < 0.5]
        print(f"{name}: Score>=0.8 정확도={high[correct_col].mean():.3f}({len(high)}개), "
              f"Score<0.5 정확도={low[correct_col].mean():.3f}({len(low)}개)")

    # 클래스별 성능
    print("\n=== 클래스별 성능 ===")
    print(f"{'Class':<6} {'PR_Acc':<8} {'PR_Score':<10} {'NG_Acc':<8} {'NG_Score':<10} {'CO_Acc':<8} {'CO_Score':<10}")
    print("-" * 70)
    for digit in range(10):
        digit_df = df[df['true_label'] == digit]
        if len(digit_df) > 0:
            print(f"{digit:<6} "
                  f"{digit_df['pr_correct'].mean():<8.3f} {digit_df['pr_score'].mean():<10.3f} "
                  f"{digit_df['ng_correct'].mean():<8.3f} {digit_df['ng_score'].mean():<10.3f} "
                  f"{digit_df['co_correct'].mean():<8.3f} {digit_df['co_score'].mean():<10.3f}")

    return df


if __name__ == "__main__":
    # 평가 실행
    results = evaluate(n_samples_eval=100)

    # 결과 분석
    df = analyze_results(results)

    # 결과 저장
    os.makedirs('llama/results', exist_ok=True)
    existing_files = glob.glob('llama/results/llama_score_results_*.csv')
    if existing_files:
        numbers = [int(f.split('_')[-1].replace('.csv', '')) for f in existing_files]
        next_num = max(numbers) + 1
    else:
        next_num = 1
    
    filename = f'llama/results/llama_score_results_{next_num}.csv'
    df.to_csv(filename, index=False)
    print(f"\n결과가 {filename}에 저장되었습니다.")
