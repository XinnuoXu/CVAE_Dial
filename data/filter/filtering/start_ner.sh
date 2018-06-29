#!/bin/bash

RUNNING=`ps ax | awk '/stanford-n[e]r/ { print $1 }'`
if [ -n "$RUNNING" ]; then
    echo "NER running ($RUNNING). Trying to kill and restart."
    kill -9 $RUNNING
    sleep 2
fi
nohup java -mx1000m -cp stanford-ner-2016-10-31/stanford-ner.jar edu.stanford.nlp.ie.NERServer -loadClassifier stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz -port 8080 -outputFormat inlineXML > ner.log 2>&1 &
