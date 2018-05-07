#python translate.py -model dialogue-model_acc_37.49_ppl_34.85_e10.pt -src data/test.en -tgt data/test.vi -report_bleu -report_rouge -verbose -gpu 0

TEST_FILE="../../test_dataset/"
MODEL=dialogue-model_acc_37.56_ppl_89.30_e26.pt

python translate.py -model $MODEL -src $TEST_FILE/test.5000.en -tgt $TEST_FILE/test.5000.vi -report_bleu -report_rouge -verbose -gpu 0
