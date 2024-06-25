from models.clip_model2 import baseline
import torch

def build_model(cfg=None, args=None, model_log=False, ):
    return baseline(model_log=model_log,
                    num_class=cfg.DATASET.NUM_CLASS,
                    num_clip=cfg.DATASET.NUM_CLIP,
                    dim_embedding=cfg.MODEL.DIM_EMBEDDING,
                    pretrain=cfg.MODEL.PRETRAIN,
                    dropout=cfg.TRAIN.DROPOUT,
                    feature_encoder=args.backbone,
                    use_TE=cfg.MODEL.TRANSFORMER,
                    multi_TE=args.multi_TE,
                    use_sim=args.use_sim,
                    use_text=args.use_text,
                    use_neg_text=args.use_neg_text,
                    use_gumbel=args.use_gumbel,
                    sim_dist=args.sim_dist,
                    concat_sim=args.concat_sim,
                    add_sim=args.add_sim,
                    use_SeqAlign=cfg.MODEL.ALIGNMENT,
                    device=args.local_rank,
                    freeze_backbone=args.freeze_backbone,
                    freeze_BN=args.freeze_BN,)

def update_logit_scale(model):
    try:
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
        model.module.logit_scale2.data = torch.clamp(model.module.logit_scale2.data, 0, 4.6052)
    except:
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
        model.logit_scale2.data = torch.clamp(model.logit_scale2.data, 0, 4.6052)
    return model
