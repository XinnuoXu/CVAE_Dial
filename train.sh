#python train.py -data data/dialogue -save_model dialogue-model -epochs 30 -report_every 100 -batch_size 16 -dropout 0.2 -src_word_vec_size 128 -tgt_word_vec_size 128 -rnn_size 128 -global_attention general -gpuid 0 -input_feed 0
python3 train.py -data data/dialogue -save_model dialogue-model -epochs 30 -report_every 100 -batch_size 256 -dropout 0.2 -src_word_vec_size 128 -tgt_word_vec_size 128 -rnn_size 128 -global_attention general -input_feed 0 -disc_model best_model/model_best.tar.gz -learning_rate 1 -gpuid 0

 
