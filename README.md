# CVAE_dialogue_generator

This project is a pytorch implementation for my paper "Xu, Dusek, Konstas, Rieser. Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity", which sadly has been neither accpted by any conference nor put on the arxiv :(


## Requirements
* Python2.7
* [GloVe model](https://github.com/maciejkula/glove-python)
* [Opensubtitles processing tool](https://github.com/WattSocialBot/movie_tools)

## Quickstart
### Step1: Download the OpenSubtitles dataset
This code is based on OpenSubtitles dataset [Automatic Turn Segmentation for Movie & TV Subtitles](http://www.diva-portal.org/smash/get/diva2:1034694/FULLTEXT01.pdf). To get the data, please contact the authors [Pierre Lison](https://github.com/plison). You should unzip the datset and name it as `opensubtitles` and put it in 

```
data/filter/
```

### Step2: Creat datasets for generator and discriminator

For the generator, a training pair consists of a dialogue context and a corresponding response. We consider three consecutive turns as the dialogue context and the following turn as the response. For the discriminator, positive examples are dialogue contexts with their following turn as the response, while negative examples are dialogue contexts with an utterance randomly sampled in the same dialogue as the response.

We use toolkit [Opensubtitles processing tool](https://github.com/WattSocialBot/movie_tools) owned by [Ondrej Dusek](https://github.com/tuetschek) to extract dialogues from OpenSubtitles dataset `data/filter/opensubtitles/`.

```
~/data/movie_tools/convert_nrno_subs.py -D -s -S train:train-dev:dev:test -r 97:1:1:1 -d all_dialogues_cased opensubtitles/ dial.jsons.txt
```
The outputs are 

* `train.dial.jsons.txt`
* `train-dev.dial.jsons.txt`
* `dev.dial.jsons.txt`
* `test.dial.jsons.txt`

as the split ratio `97:1:1:1` with format of one dialogue per line

```
["utterance 1", "utterance 2", "utterance 3"...]
```

for example,

```
["Watch out !", "Oh , what fun !", "JON :", "That was fun .", "Oh , that was great !", "Oh , time for a break ?", "Dad , I 'm hungry .", "I 'm really hungry .", "Can we eat now ?", "Keep your shirt on .", "We 'll be in Potter 's Cove in 20 minutes .", "OK , how about some pictures ?", "Here we go .", "Everybody smile .", "Say cheese ."]
```

Then, we construct the training dataset for generator and discriminator from `train.dial.jsons.txt` by running

```
python data_reading.py
```

The outputs are

* `train.en` inputs of encoder in generator (dialogue contexts)
* `train.vi` outputs of decoder in generator (expacted responses)
* `train.pos` positive examples for discriminator (dialogue contexts with their following turn as the response)
* `train.neg` negative examples for discriminator (dialogue contexts with an utterance randomly sampled in the same dialogue as the response)

The format of `train.en` is `utterance1 <u2> utterance2 <u1> utterance3` in each line, for example

```
well , i 'm glad you called me . <u2> i 'm not . <u1> no , you did the right thing .
```

The format of `train.vi` is `response` in each line, for example

```
you 'll protect him , won 't you ?
```

The formats for `train.pos` and `train.neg` are the same `utterance1 <u2> utterance2 <u1> utterance3 \t response`, for example

```
pull up sooner . <u2> ok , skipper ! <u1> do you think they 'll ever get it ?	     give them a week .
```

At last, we randomly sample 5000 cases for `train-dev, dev, test` separately by running following commands and outputs for each set are similar with training set.
* `python data_reading_shaffle.py ` for train-dev set
* `python data_reading_shaffle.py dev` for dev set
* `python data_reading_shaffle.py test` for testing set


### Step3: Filter the training set for generator

#### Step3.1: Train GloVe model on OpenSubtitles

Run the following command in `data/filter/` to read subtitles from json files and save in file `bag_of_words` in the same directory.
```
python read_html.py
```
Then, run the following two commands to train a [GloVe model](https://github.com/maciejkula/glove-python) on the OpenSubtitles dataset. `get_corpus.py` is used to build the corpus model `corpus.model` and `train.py` train the model on `corpus.model`. The trained model is `glove.model` in the same directory.
```
python get_corpus.py
python train.py
```

#### Step3.2: Filter the training set for generator

```
python get_glove_score.py train
```
The outputs for this command is cosine distance of the two semantic vectors of a dialogue context and its response (Eq.1 in the paper). The format is `cosine distance \t dialogue context \t response`. For example

```
0.9228650507713863	  they tell the whole story . <u2> i sent them , but i want the weekend . <u1> please , mr president .	    only at the weekend .
```
Then you can filter training pairs with lower coherence score (cosine distance) and rewrite the `train.en` file with the filtered dialogue contexts and `train.vi` file with their responses.

### Step4: Training for Generator (and Discriminator)

#### Step4.1: Data copying

You need to copy the following data from `data/filter/` to `data/`.

* `train.en`
* `train.vi`
* `dev.en`
* `dev.vi`
* `test.en`
* `test.vi`

#### Step4.2: Dictionary building and data preparation

```
sh preprocess.sh
```

This command will create three socuments in `data/`.

* `dialogue.train.1.pt`
* `dialogue.valid.1.pt`
* `dialogue.vocab.pt`

#### Step4.3: Discriminator training

```
cd get_c/
sh train.sh
```

This command will create `glove.model` in `get_c/`.

#### Step4.4: Generator training

Now, go back to the main directory. Run the following command to train the CVAEf_CGate generator

```
python train.py -data data/dialogue -save_model dialogue-model -epochs 30 -report_every 100 -batch_size 128 -dropout 0.2 -src_word_vec_size 128 -tgt_word_vec_size 128 -rnn_size 128 -global_attention general -input_feed 0 -glove_dir get_c/glove.model -learning_rate 1 -context_gate both
```

The trained models are named as `dialogue-model_acc_*`

### Step5: Inference

```
python translate.py -model $MODEL -src data/test.en -tgt data/test.vi -report_bleu -verbose
```

The predictions are saved in file `pred.txt`
