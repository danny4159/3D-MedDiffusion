# MR to CT Translation using ControlNet - 완전 가이드

본 가이드는 3D-MedDiffusion 프레임워크를 사용하여 MR to CT translation을 수행하는 전체 파이프라인을 설명합니다.

---

## 핵심 개념

### MR to CT Translation의 원리

1. **BiFlowNet (LDM)**: **CT latent의 분포를 학습**
   - CT 이미지를 생성할 수 있는 prior 모델
   - **CT latent만으로 학습** (MR은 사용하지 않음!)

2. **ControlNet**: **MR을 condition으로 받아 BiFlowNet을 제어**
   - BiFlowNet이 학습한 CT 분포 내에서
   - MR에 corresponding한 CT를 생성하도록 가이드

### 왜 CT만 BiFlowNet으로 학습하나?

- **목적**: MR → CT translation (CT 생성)
- **BiFlowNet의 역할**: CT를 생성하는 방법을 학습
- **ControlNet의 역할**: MR 정보를 받아서 "어떤" CT를 생성할지 결정
- **결론**: BiFlowNet은 CT의 분포만 알면 되므로 **CT만 학습**

---

## 전체 학습 파이프라인

```
1. PatchVolume AE 학습 (MR + CT)
   └─> MR과 CT를 latent space로 압축하는 방법 학습

2. Latent 생성
   ├─> MR images → MR latents (ControlNet condition용)
   └─> CT images → CT latents (BiFlowNet 학습용 + ControlNet target용)

3. BiFlowNet 학습 (CT latents만)
   └─> CT latent의 분포를 학습 (unconditional 또는 single-class)

4. ControlNet 학습 (MR latents + CT latents)
   └─> MR condition을 받아 BiFlowNet이 올바른 CT를 생성하도록 학습

5. Inference
   └─> 새로운 MR → ControlNet + BiFlowNet → CT 생성
```

---

## 전제 조건

**필수 요구사항**:
1. ✅ **Paired MR-CT 데이터셋**
   - 동일 환자의 MR과 CT가 paired로 존재
   - Filename이 정확히 매칭 (예: `patient001.nii.gz`)

---

## Step-by-Step 가이드

### Step 0: PatchVolume AutoEncoder 학습 (이미 완료 가정)

MR과 CT 모두 포함된 데이터로 PatchVolume AE를 학습했다고 가정합니다.

```bash
# 이미 완료:
# results/PatchVolume_8x_stage2/my_model/version_X/checkpoints/latest_checkpoint.ckpt
```

---

### Step 1: Paired MR-CT 데이터 준비

디렉토리 구조:
```
data/
├── MR_images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── ...
└── CT_images/
    ├── patient001.nii.gz  # ⚠️ MR과 동일한 filename!
    ├── patient002.nii.gz
    └── ...
```

**중요**: MR과 CT 파일명이 정확히 일치해야 paired dataset으로 인식됩니다!

---

### Step 2: MR과 CT를 Latent로 인코딩

#### 2-1. CT Latents 생성 (BiFlowNet 학습용)

**JSON 파일 생성** (`config/CT_data.json`):
```json
{
    "CT": "data/CT_images"
}
```

**Latent 생성**:
```bash
python train/generate_training_latent.py \
  --data-path config/CT_data.json \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --batch-size 2 \
  --num-workers 4
```

**출력**: `data/CT_images_latents/` 디렉토리 생성

#### 2-2. MR Latents 생성 (ControlNet condition용)

**JSON 파일 생성** (`config/MR_data.json`):
```json
{
    "MR": "data/MR_images"
}
```

**Latent 생성**:
```bash
python train/generate_training_latent.py \
  --data-path config/MR_data.json \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --batch-size 2 \
  --num-workers 4
```

**출력**: `data/MR_images_latents/` 디렉토리 생성

---

### Step 3: BiFlowNet 학습 (CT latents만)

#### BiFlowNet 학습용 JSON 생성 (`config/CT_only_dataset.json`):
```json
{
    "0": "data/CT_images_latents"
}
```

#### BiFlowNet 학습 (단일 클래스, CT만)

```bash
# 8 GPU 학습 예시
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29513 \
  train/train_BiFlowNet_SingleRes.py \
  --data-path config/CT_only_dataset.json \
  --results-dir results/BiFlowNet_CT_only \
  --num-classes 1 \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --resolution 16 64 64 \
  --batch-size 48 \
  --num-workers 48 \
  --epochs 800
```

**핵심**: `--num-classes 1` → CT만 학습 (unconditional or single-class)

**출력**:
- `results/BiFlowNet_CT_only/XXX-BiFlowNet/checkpoints/XXXXXXX.pt`

---

### Step 4: ControlNet 학습 (MR → CT)

#### 단일 GPU 학습

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29517 \
  train/train_MR2CT_ControlNet.py \
  --mr-latent-path data/MR_images_latents \
  --ct-latent-path data/CT_images_latents \
  --results-dir results/MR2CT_ControlNet \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --ldm-ckpt results/BiFlowNet_CT_only/XXX-BiFlowNet/checkpoints/XXXXXXX.pt \
  --num-classes 1 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 1000 \
  --log-every 50 \
  --ckpt-every 500 \
  --resolution 16 64 64 \
  --filter-by-resolution \
  --lr 1e-4 \
  --control-scale 1.0
```

#### 다중 GPU 학습 (예: 2 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29517 \
  train/train_MR2CT_ControlNet.py \
  --mr-latent-path data/MR_images_latents \
  --ct-latent-path data/CT_images_latents \
  --results-dir results/MR2CT_ControlNet \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --ldm-ckpt results/BiFlowNet_CT_only/XXX-BiFlowNet/checkpoints/XXXXXXX.pt \
  --num-classes 1 \
  --batch-size 8 \
  --num-workers 8 \
  --epochs 1000 \
  --log-every 50 \
  --ckpt-every 500 \
  --resolution 16 64 64 \
  --filter-by-resolution
```

**핵심 파라미터**:
- `--num-classes 1`: BiFlowNet에서 사용한 클래스 수와 동일
- `--control-scale`: ControlNet 영향력 (1.0 = 기본, 더 크면 MR condition 영향 증가)
- `--filter-by-resolution`: Latent shape이 일치하는 샘플만 사용

---

### Step 5: TensorBoard 모니터링

```bash
tensorboard --logdir results/MR2CT_ControlNet
```

**주요 메트릭**:
- `train/loss`: Training loss (감소 추이 확인)
- `val/MR_input`: Input MR 이미지 (고정 샘플)
- `val/CT_ground_truth`: Ground truth CT 이미지
- `val/CT_predicted`: 생성된 CT 이미지
- `val/MR2CT_comparison`: MR | GT CT | Pred CT 비교

**학습이 잘 되고 있는지 확인**:
- Loss가 점진적으로 감소
- Validation 이미지에서 CT가 점점 명확하게 생성됨
- MR의 구조가 CT에서 유지되면서 intensity가 CT로 변환됨

---

### Step 6: Inference (MR → CT Translation)

#### 단일 MR 이미지 변환

```bash
python evaluation/MR2CT_translation.py \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --ldm-ckpt results/BiFlowNet_CT_only/XXX-BiFlowNet/checkpoints/XXXXXXX.pt \
  --controlnet-ckpt results/MR2CT_ControlNet/000-MR2CT-Controlnet/checkpoints/0010000.pt \
  --input-mr data/test_MR/patient_test.nii.gz \
  --output-dir results/translated_CT \
  --num-classes 1 \
  --fixed-seed \
  --seed 42
```

#### 여러 MR 이미지 일괄 변환

```bash
python evaluation/MR2CT_translation.py \
  --AE-ckpt results/PatchVolume_8x_stage2/my_model/version_9/checkpoints/latest_checkpoint.ckpt \
  --ldm-ckpt results/BiFlowNet_CT_only/XXX-BiFlowNet/checkpoints/XXXXXXX.pt \
  --controlnet-ckpt results/MR2CT_ControlNet/000-MR2CT-Controlnet/checkpoints/0010000.pt \
  --input-mr data/test_MR \
  --output-dir results/translated_CT \
  --num-classes 1 \
  --fixed-seed \
  --seed 42
```

**출력**: `results/translated_CT/patient_XXX_translated_CT.nii.gz`

---

## 주요 파라미터 설명

### BiFlowNet 학습

- `--num-classes 1`: **CT만 학습** (unconditional/single-class)
- `--data-path`: CT latents JSON 파일 경로
- `--resolution`: Latent resolution (8x downsampling → [16, 64, 64])
- `--batch-size`: Per-GPU batch size
- `--epochs`: 학습 epoch 수 (800 권장)

### ControlNet 학습

- `--mr-latent-path`: MR latent 디렉토리 (condition)
- `--ct-latent-path`: CT latent 디렉토리 (ground truth)
- `--num-classes 1`: BiFlowNet과 동일하게 1
- `--control-scale`: ControlNet 영향력 (0.5~2.0)
  - 1.0: 기본값
  - > 1.0: MR condition 영향 증가 (더 faithful to MR)
  - < 1.0: MR condition 영향 감소 (더 creative)
- `--lr`: Learning rate (1e-4 기본값)
- `--filter-by-resolution`: Latent shape 필터링

### Inference

- `--fixed-seed`: 재현 가능한 결과를 위해 seed 고정
- `--seed`: Random seed 값
- `--control-scale`: ControlNet 영향력 조절 (학습 시와 동일 권장)

---

## 파일 구조 (학습 완료 후)

```
3D-MedDiffusion/
├── data/
│   ├── MR_images/               # Original MR images
│   ├── CT_images/               # Original CT images (paired)
│   ├── MR_images_latents/       # MR latents (condition)
│   └── CT_images_latents/       # CT latents (BiFlowNet 학습 + ControlNet target)
├── config/
│   ├── MR_data.json             # MR latent 생성용
│   ├── CT_data.json             # CT latent 생성용
│   └── CT_only_dataset.json     # BiFlowNet 학습용
├── results/
│   ├── PatchVolume_8x_stage2/   # Trained AutoEncoder
│   ├── BiFlowNet_CT_only/       # Trained BiFlowNet (CT only)
│   └── MR2CT_ControlNet/
│       └── 000-MR2CT-Controlnet/
│           ├── checkpoints/      # ControlNet checkpoints
│           ├── samples/          # Validation samples
│           └── events.out.tfevents.*
├── dataset/
│   └── MR2CT_dataset.py         # MR-CT paired dataset
├── train/
│   └── train_MR2CT_ControlNet.py
└── evaluation/
    └── MR2CT_translation.py
```

---

## 트러블슈팅

### 1. **Paired data 매칭 안 됨**
```
Warning: No matching CT latent for patient001.nii.gz
```
**해결**: MR과 CT 파일명이 정확히 일치하는지 확인

### 2. **Latent 생성 시 OOM**
```bash
# batch-size를 1로 줄이세요
python train/generate_training_latent.py --batch-size 1 ...
```

### 3. **ControlNet Loss가 수렴하지 않음**
- `--control-scale` 조정 (0.5, 1.0, 1.5, 2.0 테스트)
- Learning rate 감소 (`--lr 5e-5` 또는 `1e-5`)
- BiFlowNet이 충분히 학습되었는지 확인 (샘플 생성 품질 체크)
- Paired data의 alignment 확인 (MR과 CT가 실제로 대응되는지)

### 4. **생성된 CT 품질이 낮음**
- 더 오래 학습 (10k+ steps 권장)
- `--control-scale` 증가 (1.5 또는 2.0)
- BiFlowNet 학습 품질 확인 (CT unconditional 생성 테스트)
- PatchVolume AE reconstruction 품질 확인

### 5. **num_classes 불일치 에러**
```
AssertionError: must specify y if and only if the model is class-conditional
```
**해결**: BiFlowNet과 ControlNet의 `--num-classes`를 동일하게 설정 (둘 다 1)

---

## 핵심 포인트 요약

### ✅ 올바른 학습 파이프라인

1. **PatchVolume AE**: MR + CT 모두 학습 ✅
2. **BiFlowNet**: **CT latent만** 학습 ✅ (num-classes=1)
3. **ControlNet**: MR latent (condition) + CT latent (target) 학습 ✅

### ✅ Class Label 설정

- **BiFlowNet**: `num-classes=1` (CT만)
- **ControlNet**: `num-classes=1` (BiFlowNet과 동일)
- **Dataset y label**: `0` (single-class, 0-indexed)

### ✅ Paired Data 요구사항

- MR과 CT 파일명 정확히 일치
- 동일 환자/스캔의 MR-CT pair
- Spatial alignment 권장 (registration)

### ✅ 모니터링 포인트

- TensorBoard에서 고정된 validation sample로 진행도 추적
- Loss 감소 추이
- 생성된 CT의 구조적 일관성
- MR의 anatomy가 CT에 올바르게 반영되는지

---

## FAQ

**Q1: BiFlowNet을 MR과 CT 둘 다로 학습시키면 안 되나요?**
A: 가능하지만 비효율적입니다. MR to CT translation에서는 CT만 생성하므로, BiFlowNet이 CT 분포만 학습하면 충분합니다. MR은 ControlNet의 condition으로만 사용됩니다.

**Q2: 기존에 MR+CT로 학습한 BiFlowNet(num-classes=2)을 사용해도 되나요?**
A: 가능합니다. 이 경우:
- ControlNet 학습 시 `--num-classes 2`
- Dataset y label을 CT의 class index로 설정 (예: 1)
- 더 범용적이지만, CT 전용 모델보다 성능이 약간 낮을 수 있음

**Q3: Control scale은 어떻게 설정하나요?**
A: 기본값 1.0부터 시작. TensorBoard에서 validation 결과를 보고:
- MR 구조가 잘 반영 안 됨 → 1.5~2.0 증가
- 너무 MR에 overfitting → 0.5~0.8 감소

**Q4: 학습 시간은 얼마나 걸리나요?**
A:
- BiFlowNet (8 GPUs): ~2-3일 (800 epochs)
- ControlNet (1 GPU): ~1-2일 (10k steps)
- 데이터셋 크기와 GPU 성능에 따라 변동

---

## Citation

이 프레임워크를 사용하시면 다음을 인용해주세요:
```
@article{3dmedddiffusion,
  title={3D-MedDiffusion: A 3D Medical Latent Diffusion Model for Controllable and High-Quality Medical Image Generation},
  ...
}
```

---

**버전**: 2.0 (CT-only BiFlowNet)
**최종 업데이트**: 2026-01-08
