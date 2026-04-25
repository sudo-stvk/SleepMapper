# SleepMapper context

- Task: binary sleep apnea classification from 30s audio clips
- Models: ResNet-18 (spectrograms), BiLSTM (MFCC), wav2vec2-tiny (raw audio)
- Dataset: Kaggle snoring + ESC-50 + PhysioNet Apnea-ECG
- Threshold: 0.35 (sensitivity-prioritized)
- Primary metric: AUC-ROC
- Deployment target: ONNX → mobile inference
- No cloud audio storage — privacy-first