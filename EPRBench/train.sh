CUDA_VISIBLE_DEVICES=2 python train.py --dataset_folder /media/amax/xiao_20T1/EventVPR \
    --foundation_model_path /media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/xiongxingxing/CricaVPR-main/dinov2_vitb14_pretrain.pth \
    --text_folder /media/amax/xiao_20T1/EventVPR/scene_descriptions \
    --use_text --lambda_contrast 0.25 --temperature 0.07 