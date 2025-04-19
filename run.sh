# Origin Dataset
python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets origin \
    --batch_size 32 \
    --model_name resnet18 
    # --checkpoint checkpoints/model_resnet18.pth \
    # --preprocessing precomputed

python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets origin \
    --batch_size 32 \
    --model_name efficientnet_b0 
    # --checkpoint ./checkpoints/model_efficientnet_b0.pth \
    # --preprocessing precomputed

python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets origin \
    --batch_size 32 \
    --model_name vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k 
    # --checkpoint ./checkpoints/model_best_vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1.pth \
    # --preprocessing precomputed

# Modify Dataset
python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets modify \
    --batch_size 32 \
    --model_name efficientnet_b0 
    # --checkpoint ./checkpoints/model_efficientnet_b0.pth \
    # --preprocessing precomputed
    
python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets modify \
    --batch_size 32 \
    --model_name resnet18 
    # --checkpoint checkpoints/model_resnet18.pth \
    # --preprocessing precomputed

python3 main.py \
    --mode train \
    --epochs 5 \
    --datasets modify \
    --batch_size 32 \
    --model_name vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k 
    # --checkpoint ./checkpoints/model_best_vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1.pth \
    # --preprocessing precomputed

# Inference
python3 inference.py \
  --tta \
  --output my_submission.csv
