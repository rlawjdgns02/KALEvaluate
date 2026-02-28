"""
FActScore 테스트 - ollama 버전
데이터셋으로 평가
"""
import os
import sys
import json

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factscore.factscorer import FactScorer

def load_data(data_path, n_samples=None):
    """JSONL 데이터 로드"""
    topics = []
    generations = []

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dp = json.loads(line)
            topics.append(dp["topic"])
            generations.append(dp["output"])

            if n_samples and len(topics) >= n_samples:
                break

    return topics, generations


def main():
    # 데이터 경로
    data_dir = os.path.join(PROJECT_ROOT, "data", "data_factscore", "labeled")

    # 어떤 모델 결과를 평가할지 선택
    # - ChatGPT.jsonl
    # - InstructGPT.jsonl
    # - PerplexityAI.jsonl
    data_file = "ChatGPT.jsonl"
    data_path = os.path.join(data_dir, data_file)

    # 샘플 수 (전체 하려면 None)
    n_samples = 5  # 테스트용으로 5개만

    print("=" * 60)
    print(f"FActScore 평가 - {data_file}")
    print("=" * 60)

    # 데이터 로드
    topics, generations = load_data(data_path, n_samples)
    print(f"로드된 샘플 수: {len(topics)}")

    for i, (topic, gen) in enumerate(zip(topics, generations)):
        print(f"\n[{i+1}] {topic}")
        print(f"    {gen[:100]}...")

    print("\n" + "=" * 60)

    # FactScorer 초기화
    fs = FactScorer(
        model_name="llama3.2",
        data_dir=PROJECT_ROOT,
        cache_dir=os.path.join(PROJECT_ROOT, ".cache", "factscore")
    )

    # 스코어 계산
    print("\n스코어 계산 중... (시간이 걸릴 수 있음)")
    result = fs.get_score(
        topics=topics,
        generations=generations,
        gamma=10,
        verbose=True
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("전체 결과")
    print("=" * 60)
    print(f"FActScore: {result['score'] * 100:.1f}%")
    if "init_score" in result:
        print(f"FActScore (length penalty 제외): {result['init_score'] * 100:.1f}%")
    print(f"응답 비율: {result['respond_ratio'] * 100:.1f}%")
    print(f"응답당 atomic facts 수: {result['num_facts_per_response']:.1f}")

    # 개별 결과
    print("\n" + "=" * 60)
    print("개별 결과")
    print("=" * 60)
    for i, (topic, decisions) in enumerate(zip(topics, result['decisions'])):
        if decisions:
            supported = sum(1 for d in decisions if d['is_supported'])
            total = len(decisions)
            score = supported / total * 100
            print(f"[{i+1}] {topic}: {score:.1f}% ({supported}/{total} facts)")
        else:
            print(f"[{i+1}] {topic}: 응답 없음")

    # 결과 저장
    output_path = os.path.join(PROJECT_ROOT, "factscore_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "data_file": data_file,
            "n_samples": len(topics),
            "score": result['score'],
            "init_score": result.get('init_score'),
            "respond_ratio": result['respond_ratio'],
            "num_facts_per_response": result['num_facts_per_response'],
            "decisions": result['decisions']
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장됨: {output_path}")

    return result


if __name__ == "__main__":
    result = main()