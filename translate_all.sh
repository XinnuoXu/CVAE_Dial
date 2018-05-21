TEST_OSA="../OSA_data.mini/"
TEST_OSC="../OSC_data.mini/"
MODEL=dialogue-model/dialogue-model_acc_38.76_ppl_33.97_e30.pt

python translate.py -model $MODEL -src $TEST_OSC/test.en -tgt $TEST_OSC/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.8
mv pred.txt pred_osc_0.8.txt

python translate.py -model $MODEL -src $TEST_OSA/test.en -tgt $TEST_OSA/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.8
mv pred.txt pred_osa_0.8.txt

python translate.py -model $MODEL -src $TEST_OSC/test.en -tgt $TEST_OSC/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.6
mv pred.txt pred_osc_0.6.txt

python translate.py -model $MODEL -src $TEST_OSA/test.en -tgt $TEST_OSA/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.6
mv pred.txt pred_osa_0.6.txt

python translate.py -model $MODEL -src $TEST_OSC/test.en -tgt $TEST_OSC/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.4
mv pred.txt pred_osc_0.4.txt

python translate.py -model $MODEL -src $TEST_OSA/test.en -tgt $TEST_OSA/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.4
mv pred.txt pred_osa_0.4.txt

python translate.py -model $MODEL -src $TEST_OSC/test.en -tgt $TEST_OSC/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.2
mv pred.txt pred_osc_0.2.txt

python translate.py -model $MODEL -src $TEST_OSA/test.en -tgt $TEST_OSA/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.2
mv pred.txt pred_osa_0.2.txt

python translate.py -model $MODEL -src $TEST_OSC/test.en -tgt $TEST_OSC/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.0
mv pred.txt pred_osc_0.0.txt

python translate.py -model $MODEL -src $TEST_OSA/test.en -tgt $TEST_OSA/test.vi -report_bleu -verbose -gpu 0 -batch_size 128 -c_control 0.0
mv pred.txt pred_osa_0.0.txt

