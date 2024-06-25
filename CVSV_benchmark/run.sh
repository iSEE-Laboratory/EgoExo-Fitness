# Exo_pairs
CUDA_VISIBLE_DEVICES=6,7 python train.py --config configs/Exo_Only/train_config_egofbau_CAT_pair_ClsSeq.yml --pair &

# Exo_pairs + Ego_pairs + Ego_Exo_pairs
CUDA_VISIBLE_DEVICES=4,5 python train.py --config configs/All_Pairs/train_config_egofbau_CAT_pair_ClsSeq.yml --pair &

# Exo_pairs + Ego_pairs + Ego_Exo_pairs
CUDA_VISIBLE_DEVICES=5,4 python train.py --config configs/No_Cross_View/train_config_egofbau_CAT_pair_ClsSeq.yml --pair &

# Ego_pairs
CUDA_VISIBLE_DEVICES=4,5 python train.py --config configs/Ego_Only/train_config_egofbau_CAT_pair_ClsSeq.yml --pair &

# Ego_Exo_pairs
CUDA_VISIBLE_DEVICES=5,4 python train.py --config configs/Cross_View_Only/train_config_egofbau_CAT_pair_ClsSeq.yml --pair &