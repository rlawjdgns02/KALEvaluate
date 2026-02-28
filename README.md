<div align="center">

# KALEval

**AI 모델 신뢰성 평가 프레임워크**

*AI 모델의 신뢰성(Trustworthiness), 강건성(Robustness), 사실성(Factuality)을 평가합니다*

</div>

---

## 연구 개요

항공 정비(MRO) 도메인에서 AI 모델 적용 시 요구되는 **모델 신뢰성 및 강건성 평가 지표 개발**을 위한 초기 분석 및 적용 연구입니다.

### 연구 목표

> *"AI 모델이 실제 운영 환경에서 얼마나 신뢰할 수 있는가?"*

- 다양한 평가 지표 조사 및 실제 모델 적용
- 평가 방법론의 유효성 검증
- 도메인 특화 평가 기준 수립을 위한 기초 연구

> **Note**: 본 프로젝트는 연구 진행 중이며, 평가 지표 및 모델 실험 결과가 지속적으로 추가될 예정입니다.

---

## 평가 지표

### 1. 강건성 평가 (Robustness) - SCORE

멀티모달 AI 모델의 일관성과 안정성을 3가지 관점에서 평가합니다.

| 지표 | 설명 | 측정 방법 |
|:-----|:-----|:----------|
| **Prompt Robustness (PR)** | 프롬프트 변화에 대한 저항성 | 10가지 다른 프롬프트로 동일 입력 평가 |
| **Non-Greedy Robustness (NG)** | 생성 과정의 안정성 | temperature=0.7로 5회 반복 샘플링 |
| **Choice Order Robustness (CO)** | 선택지 순서 편향 | 5가지 선택지 순서 변형으로 평가 |

### 2. 사실성 평가 (Factuality) - FActScore

LLM이 생성한 텍스트의 사실적 정확성을 검증합니다.

```
생성 텍스트 → 원자적 사실 분해 → Wikipedia 검증 → 지원 비율 산출
```

- 생성 텍스트를 **원자적 사실(Atomic Facts)**로 분해
- Wikipedia 지식베이스를 통해 각 사실을 검증
- 지원되는 사실의 비율로 최종 점수 산출

### 3. 지식 평가 - MMLU

57개 주제에 대한 다중선택 문제를 통해 모델의 지식 수준을 평가합니다.

### 4. 불균형 데이터 영향 분석

데이터 불균형이 모델 성능에 미치는 영향을 6가지 시나리오로 체계적 분석합니다.

---

## 진행 상황

### 완료

- [x] SCORE 강건성 평가 구현 (LLaMA Vision 기반)
- [x] FActScore 사실성 평가 구현 (Ollama 연동)
- [x] MMLU 지식 벤치마크 평가 구현
- [x] 불균형 데이터셋 실험 (6가지 시나리오)

### 진행 중

- [ ] 추가 평가 지표 조사 및 적용
- [ ] 다양한 모델에 대한 실험 결과 축적
- [ ] 도메인 특화 평가 기준 정립

---

## 참고 문헌

- FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation (Min et al., 2023)
- MMLU: Measuring Massive Multitask Language Understanding (Hendrycks et al., 2021)
- SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models

---

## License

This project is for research purposes.
