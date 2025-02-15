# RNN-LSTM_proj
Implement RNN and LSTM only with numpy. We also compare the performance according to the differences in each structure, optimizer, embedding size, etc.

## 개요
이 프로젝트는 RNN 및 LSTM을 활용하여 텍스트 데이터를 기반으로 이모지를 분류하는 모델을 구현하고 비교 분석하는 것을 목표로 합니다. 또한, 다양한 하이퍼파라미터(optimizer, embedding 크기, dropout 등)가 모델 성능에 미치는 영향을 평가합니다.

## 모델 및 실험 구성
다양한 모델 및 학습 조건을 비교하여 성능을 분석하였습니다.

| 모델 | Optimizer | Embedding 크기 | Dropout | Train Loss | Test Loss | Accuracy |
|------|----------|--------------|---------|------------|-----------|----------|
| RNN | SGD | 50d | X | 0.0004 | 8.3475 | 52% |
| LSTM | SGD | 50d | X | 0.4882 | 0.7041 | 78.12% |
| LSTM | Adam | 50d | X | 0.0029 | 0.6715 | 84.38% |
| LSTM | SGD | 100d | X | 0.3110 | 0.9382 | 75% |
| LSTM | SGD | 50d | 0.7 | 0.4915 | 0.9092 | 68.75% |

## 주요 분석
1. **Optimizer 비교 (SGD vs Adam)**
   - 동일한 LSTM 모델에서 Adam을 사용했을 때, SGD보다 낮은 test loss와 높은 accuracy를 기록함.
   - Adam은 학습률을 동적으로 조정하며 안정적인 기울기 업데이트를 수행하는 것이 주요 원인.

2. **RNN vs LSTM**
   - RNN보다 LSTM이 test loss가 낮고 accuracy가 높음.
   - LSTM의 게이트 구조가 long-term dependency를 잘 처리하기 때문.

3. **Embedding 크기 비교 (50d vs 100d)**
   - 50d보다 100d를 사용했을 때 성능이 오히려 감소하는 경향.
   - Embedding 차원이 크면 정보량이 많아지지만, 데이터가 충분하지 않으면 오버피팅 발생 가능.

4. **Dropout 적용**
   - Dropout(0.7)을 적용했을 때 모델 성능이 낮아짐.
   - 과적합 방지를 위해 사용되었지만, 과도한 dropout으로 인해 학습이 충분히 이루어지지 않음.

## Word2Vec vs GloVe
- **Word2Vec**: 주변 단어의 문맥 정보를 바탕으로 신경망이 단어 벡터를 학습함 (Skip-gram, CBOW 방식).
- **GloVe**: 동시 등장 빈도 행렬을 기반으로 전체적인 단어 관계를 학습함.
- 본 프로젝트에서는 감정 분류 작업에서 전역적 통계 정보를 포함한 GloVe가 더 효과적이라 판단하여 사용함.

## 결과 시각화
각 실험에서 얻은 Loss 및 Accuracy 그래프를 포함하여 비교 분석.

(그래프 및 결과 예제 이미지 추가 가능)

## 실행 방법
1. `glove.6B.50d.txt` 또는 `glove.6B.100d.txt`를 다운로드하여 프로젝트 폴더에 배치합니다.
2. 아래 명령어를 실행하여 환경을 설정합니다.
   ```sh
   pip install -r requirements.txt
   ```
3. 모델을 학습하고 결과를 확인합니다.
   ```sh
   python train.py
   ```

## 참고 자료
- GloVe: [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/)
- Adam Optimizer: [Paper](https://arxiv.org/abs/1412.6980)
- LSTM: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
