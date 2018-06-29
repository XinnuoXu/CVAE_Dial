#!/usr/bin/env python
# -"- coding: utf-8 -"-

from __future__ import unicode_literals

import re
from argparse import ArgumentParser
import sys
from filtering.tokenize import Tokenizer
import numpy as np
import xml.etree.ElementTree as ET
import os.path
import codecs
import datetime

# Start IPdb on error in interactive mode
import sys


class Movie(object):

    def __init__(self, subtitle_file):
        self.filename = subtitle_file
        self.xml_tree = ET.parse(subtitle_file)
        self.scene_bound_time = datetime.timedelta(seconds=4)
        self.dialogues = None
        self.num_turns = 0

    def _extract_dialogues(self):
        """Extract speakers and dialogue lines from the movie. Return the dialogues
        and the total number of turns."""

        dialogues = []
        cur_dialogue = []
        cur_statement = ''
        cur_speaker = ''
        prev_start_time = - self.scene_bound_time - datetime.timedelta(seconds=1)
        prev_end_time = - self.scene_bound_time - datetime.timedelta(seconds=1)
        turns = 0

        for seg in self.xml_tree.getroot():
            # extract information from the segment
            text = ' '.join([tok.text for tok in seg if tok.tag == 'w'])
            start_time = prev_start_time
            end_time = prev_end_time
            for time_spec in seg.findall('time'):
                time_val = self._parse_time(time_spec.attrib['value'])
                if not time_val:  # invalid expression, ignore it
                    continue
                if time_spec.attrib['id'].endswith('S'):  # start time
                    start_time = time_val
                elif time_spec.attrib['id'].endswith('E'):  # end time
                    end_time = time_val

            # ignore '-' at beginning of segment
            if text.startswith('- '):
                text = text[2:]

            # detect turn boundary in manual/automatic annotation
            turn_bound = (False if (seg.attrib.get('continued') or
                                    seg.attrib.get('turn', 1) < 0.5) else True)

            # starting a new scene
            if (turn_bound and (start_time - prev_end_time) > self.scene_bound_time):
                if cur_statement:
                    cur_dialogue.append((cur_speaker, cur_statement))
                if len(cur_dialogue) > 1:
                    dialogues.append(cur_dialogue)
                    turns += len(cur_dialogue)
                cur_dialogue = []
                cur_statement = ''

            # starting a new statement
            elif turn_bound and cur_statement:
                cur_dialogue.append((cur_speaker, cur_statement))
                cur_statement = ''

            # buffering the current values
            cur_statement += (' ' if cur_statement else '') + text
            cur_speaker = seg.attrib.get('speaker', '')
            prev_start_time = start_time
            prev_end_time = end_time

        # clearing the buffers at the end
        if cur_statement:
            cur_dialogue.append((cur_speaker, cur_statement))
        if len(cur_dialogue) > 1:
            dialogues.append(cur_dialogue)
            turns += len(cur_dialogue)

        self.dialogues = dialogues
        self.num_turns = turns

    def _parse_time(self, text):
        time_expr = re.match(r'([0-9]+):([0-9]+):([0-9]+)[,.]([0-9]+)', text)
        if not time_expr:  # invalid expression
            return None
        hrs, mins, secs, millis = [int(val) for val in time_expr.groups()]
        return datetime.timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=millis)

    def _remove_bracketed(self, text):
        """Remove bracketed stuff -- remarks, emotions etc."""
        text = re.sub(r'\([^\)]*\)', r'', text)
        text = re.sub(r'\[[^\]]*\]', r'', text)
        return text

    def _postprocess(self, do_tokenize=False, do_lowercase=False):
        """Postprocess dialogues (remove bracketed stuff, tokenize, lowercase)."""

        tok = Tokenizer()
        for dialogue in self.dialogues:
            # we're changing the list so we need to use indexes here
            for turn_no in xrange(len(dialogue)):
                speaker, statement = dialogue[turn_no]

                speaker = self._remove_bracketed(speaker)
                statement = self._remove_bracketed(statement)
                if do_tokenize:
                    statement = tok.tokenize(statement)
                if do_lowercase:
                    statement = statement.lower()
                dialogue[turn_no] = (speaker, statement)  # assign new values

            # remove all turns that have been rendered empty by the postprocessing
            dialogue[:] = [(speaker, statement) for speaker, statement in dialogue
                           if statement is not None and statement.strip()]

    def get_dialogues(self, do_tokenize=False, do_lowercase=False):
        """Load and process one movie file, return the dialogues."""
        # actually extract the dialogues
        self._extract_dialogues()

        # remove brackets, tokenize, lowercase
        self._postprocess(do_tokenize, do_lowercase)


def process_all(args):

    year_from, year_to = 0, 100000
    dialogues = []

    fpout = open("./bag_of_words", "w")

    # read and process all movies
    for main_dir, subdirs, files in os.walk("./opensubtitles/"):

        if not files:
            continue

        # filter years (assume directories structured by year)
        year = re.search(r'\b[0-9]{4}\b', main_dir)
        if year:
            year = int(year.group(0))
            if year < year_from or year > year_to:
                continue

        movie_file = os.path.join(main_dir, files[0])  # just use the 1st file
        # load and try to identify the movie
        movie = Movie(movie_file)
        # extract dialogues from the movie
        movie.get_dialogues(args.tokenize, args.lowercase)
        if not movie.dialogues:
            continue
        for dialogue in movie.dialogues:
	    try:
	    	sentence = " ".join([utt for _, utt in dialogue]).decode("utf8").encode("utf8").strip()
	    	dialogues.append(sentence)
	    except:
		pass
	fpout.write(movie_file + "\t" + " ".join(dialogues) + "\n")
	del dialogues[:]

    fpout.close()

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-l', '--lowercase', action='store_true', help='Lowercase all outputs?')
    ap.add_argument('-t', '--tokenize', action='store_true', help='Tokenize all outputs?')
    ap.add_argument('-d', '--directory', type=str, default='.',
                    help='Output directory (default: current)')

    args = ap.parse_args()
    process_all(args)
