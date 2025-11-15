# train_guard_models.py
# 딱 한 번만 실행해서:
# - NLI 분류기 학습
# - 대한민국 입장 팩트 인덱스(fact_texts.json + fact_embeddings.npy) 생성

from Moderation_API import (
    train_nli_classifier,
    build_fact_index_from_positions,
)

if __name__ == "__main__":
    # 1) NLI 분류기 학습 (augmented_dataset.json 기준)
    print("[TRAIN] NLI 분류기를 학습합니다...")
    train_nli_classifier()

    # 2) 대한민국 입장 팩트 인덱스 구축 (korea_fact_positions.json → fact_texts.json + fact_embeddings.npy)
    print("[TRAIN] 대한민국 입장 팩트 인덱스를 구축합니다...")
    build_fact_index_from_positions()

    print("✅ NLI 모델 + 팩트 인덱스 준비 완료")
