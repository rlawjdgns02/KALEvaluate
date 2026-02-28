# mmlu/download_mmlu.py

from datasets import load_dataset
import os

# 저장 경로
SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mmlu')


def download_and_save_mmlu():
    """MMLU 전체 데이터셋 다운로드 및 저장"""

    print("Downloading MMLU dataset (all subjects)...")
    print(f"Save path: {SAVE_PATH}\n")

    # "all" config로 57개 주제 한번에 로드
    dataset = load_dataset("cais/mmlu", "all", trust_remote_code=True)

    print(f"Total samples:")
    print(f"  - test: {len(dataset['test'])}")
    print(f"  - dev: {len(dataset['dev'])}")
    print(f"  - validation: {len(dataset['validation'])}")

    # 저장
    print(f"\nSaving to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    dataset.save_to_disk(SAVE_PATH)

    print("Done!")
    return dataset


if __name__ == "__main__":
    download_and_save_mmlu()
