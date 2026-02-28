"""
FActScore 테스트 - llama 모델 평가
1. llama로 인물 전기 생성
2. 생성된 텍스트를 llama로 평가
"""
import os
import sys
import json
import ollama

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from factscore.factscorer import FactScorer


def generate_biography(topic, model_name="llama3.2"):
    """llama로 인물 전기 생성"""
    prompt = f"Tell me a bio of {topic}."

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 512}
    )

    return response['message']['content']


def main():
    # 평가할 인물 리스트
    topics = [
        "Albert Einstein",
        "Marie Curie",
        "Isaac Newton",
        "Nikola Tesla",
        "Leonardo da Vinci",
    ]

    # 샘플 수 조절
    n_samples = 3  # 테스트용으로 3개만
    topics = topics[:n_samples]

    print("=" * 60)
    print("FActScore 평가 - llama3.2 모델")
    print("=" * 60)

    # 1. llama로 인물 전기 생성
    print("\n[1단계] llama로 인물 전기 생성 중...")
    generations = []
    for i, topic in enumerate(topics):
        print(f"  생성 중: {topic}...")
        gen = generate_biography(topic)
        generations.append(gen)
        print(f"  [{i+1}/{len(topics)}] {topic} 완료")
        print(f"      {gen[:100]}...")

    print("\n" + "=" * 60)

    # 2. FactScorer 초기화
    print("\n[2단계] FActScore 평가 준비...")
    fs = FactScorer(
        model_name="llama3.2",
        data_dir=PROJECT_ROOT,
        cache_dir=os.path.join(PROJECT_ROOT, ".cache", "factscore"),
        abstain_detection_type="generic"
    )

    # 3. 스코어 계산
    print("\n[3단계] 스코어 계산 중... (시간이 걸릴 수 있음)")
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

    # 상세 결과 출력
    print("\n" + "=" * 60)
    print("상세 결과 (각 fact별 판정)")
    print("=" * 60)
    for i, (topic, gen, decisions) in enumerate(zip(topics, generations, result['decisions'])):
        print(f"\n[{i+1}] {topic}")
        print(f"생성된 텍스트: {gen[:200]}...")
        if decisions:
            print("판정:")
            for d in decisions:
                status = "O" if d['is_supported'] else "X"
                print(f"  {status}: {d['atom']}")

    # 결과 저장
    output_path = os.path.join(PROJECT_ROOT, "factscore_llama_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model": "llama3.2",
            "n_samples": len(topics),
            "topics": topics,
            "generations": generations,
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
