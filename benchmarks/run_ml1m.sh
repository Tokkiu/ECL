python run_recbole.py --model=ECL  --n_layers=2 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.2 --ignore_pad=True --disable_aug=h --mask_strategy=sample --share_param=all --discriminator_combine=addall --discriminator_bidirectional=False --contras_target=avgk --contras_k=20 --mask_ratio=0.2 --train_stage=alltrain --encoder_loss_weight=1 --always_con=True --contrastive_loss_weight=0.001 --discriminator_loss_weight=0.05 --generate_loss_weight=0.2 --dataset=ml-1m --config_files="conf/config_d_ml-1m.yaml conf/config_t_train.yaml conf/config_m_DCL.yaml" --gpu_id=0