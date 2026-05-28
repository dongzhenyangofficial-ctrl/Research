from models.vaanet import VAANet
from models.bett import Bett


def generate_model(opt, model_name='VAANet'):
    if model_name=='VAANet':
        model = VAANet(
            snippet_duration=opt.snippet_duration,
            sample_size=opt.sample_size,
            n_classes=opt.n_classes,
            seq_len=opt.seq_len,
            audio_embed_size=opt.audio_embed_size,
            audio_n_segments=opt.audio_n_segments,
            pretrained_resnet101_path=opt.resnet101_pretrained,
        )
    elif model_name=='Bett':
        model = Bett(
            img_size=opt.sample_size,
            num_patches=opt.seq_len*opt.snippet_duration,
            embed_dim=512,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=0,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_mean_pooling=True,
            init_values=0.1
        )
    else:
        raise Exception
    return model, model.parameters()
