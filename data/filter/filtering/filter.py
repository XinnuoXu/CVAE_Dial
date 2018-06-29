#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs
import re
import sys
from unidecode import unidecode
import ner


class Filter(object):

    def __init__(self, cfg=''):
        min_len, max_len, ne_types, profanities, _ = (cfg + '::::').split(':', 4)
        self.profanities = self._load_profanities_list(profanities) if profanities else None
        self.max_length = int(max_len) if max_len else -1
        self.min_length = int(min_len) if min_len else -1
        self.ban_punct_only = (min_len >= 0)
        if ne_types:
            self.ne_types = re.compile('^([' + ne_types + '].+)$')
            self.ner = ner.SocketNER(host='localhost', port=8080)
        else:
            self.ne_types = None


    def _load_profanities_list(self, filename):
        with codecs.open(filename, 'rb', 'UTF-8') as fh:
            profanities = fh.readlines()
        pattern = r'\b(' + '|'.join([profanity.strip() for profanity in profanities]) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def filter_sentence(self, sent):
        # normalize sentence
        sent = unidecode(sent)
        # remove URLs, HTML tags and entities, weird characters
        sent = re.sub(r'https? ?: ?/ ?/[^ ]*', '', sent)
        sent = re.sub(r'&(amp|lt|gt);', '', sent)
        sent = re.sub(r'< ?/? ?(strong|b|span|u|i|em|h[1-7]|li|ul|ol|div)(?: [^>]*)?>', '', sent)
        sent = re.sub(r'\[[^)]*\]', '', sent)  # delete all stuff in brackets
        sent = re.sub(r'\([^)]*\)', '', sent)  # delete all stuff in brackets
        sent = re.sub(r'[a-z.]*@[a-z.]*', '', sent)  # delete email adresses
        sent = re.sub(r'[^A-Za-z0-9\',;:!?.-]', ' ', sent)  # delete all but listed characters
        sent = re.sub(r' +', r' ', sent).strip()
        # sentence too long
        if self.max_length >= 0 and sent.count(' ') > self.max_length:  # TODO approximation
            return None
        # sentence too short
        if self.min_length >= 0 and sent.count(' ') < self.min_length - 1:
            return None
        # sentence contains profanities
        if self.profanities and re.search(self.profanities, sent):
            return None
        # sentence only contains punctuation characters
        if self.ban_punct_only and re.match(r'^[ \',;:!?.-]*$', sent):
            return None
        # sentence contains NEs
        if self.ne_types:
            ents = self.ner.get_entities(sent)
            if self.ne_types.match(' '.join(ents.keys())):
                return None
        return sent

if __name__ == '__main__':

    filter = Filter({'profanities_list': 'profanities.txt',
                     'max_length': 20,
                     'ban_punct_only': True})

    stdin = codecs.getreader('UTF-8')(sys.stdin)
    stdout = codecs.getreader('UTF-8')(sys.stdout)

    for line in stdin:
        res = filter.filter([line])
        if res:
            print >> stdout, res[0], "\n"
        else:
            print >> stdout, '<<REMOVED>>', "\n"
