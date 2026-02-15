current_pth=$(pwd)
fairseq_pth=$current_pth/fairseq
src_pth=$current_pth/src

data_pth=$current_pth/labels_unit/audio_visual
checkpoint_save_pth=$current_pth/exp/su2sw_pretraining/audio_visual

decoder_embed_dim=1024
decoder_ffn_embed_dim=4096
decoder_attention_heads=8
su2sw_embedding_dim=1024
decoder_layers=9


PYTHONPATH=$fairseq_pth \
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train \
    --config-dir $src_pth/conf/ \
    --config-name su2sw.yaml \
    task.data=$data_pth \
    task.label_dir=$data_pth \
    task.tokenizer_bpe_model=$current_pth/spm1000/spm_unigram1000.model \
    hydra.run.dir=$checkpoint_save_pth \
    common.user_dir=$src_pth \
    model.decoder_layers=$decoder_layers \
    model.decoder_embed_dim=$decoder_embed_dim \
    model.su2sw_embedding_dim=$su2sw_embedding_dim \
    model.decoder_ffn_embed_dim=$decoder_ffn_embed_dim \
    model.decoder_attention_heads=$decoder_attention_heads