#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspire.py
Collection of functions useful for the INSPIRE challenge.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import pprint as pp
import collections

from . import common
from . import edit_distance

UTF8_NORMALIZATION = 'NFD'


class MyPrettyPrinter(pp.PrettyPrinter):
    _escape = dict((q, dict(((c, unicode(repr(chr(c)))[1:-1])
                             for c in range(32) + [ord('\\')] +
                             range(128, 161)),
                            **{ord(q): u'\\' + q}))
                   for q in ["'", '"'])

    def format(self, obj, context, maxlevels, level):
        if type(obj) is unicode:
            q = "'" if "'" not in obj or '"' in obj \
                else '"'
            return ("u" + q + obj.translate(self._escape[q]) +
                    q, True, False)
        return pp.PrettyPrinter.format(
            self, obj, context, maxlevels, level)


def pprint(obj, **kwargs):
    printer = MyPrettyPrinter(**kwargs)
    printer.pprint(obj)


class Dataset(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        return json.dumps(self, indent=2)


class Lexicon(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, key.upper())

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key.upper(), value)

    def __contains__(self, key):
        return dict.__contains__(self, key.upper())

    def __repr__(self):
        return json.dumps(self, indent=2)


class Submission(dict):
    def __init__(self, authors=[], email='', challenge_edition='', lexicon=None):
        self.update({'metadata': {'authors': authors,
                                  'email': email,
                                  'challenge_edition': challenge_edition},
                     'lexicon': lexicon,
                     'tokens': {}})

    def where_task(self, token_id, confusion_probability):
        self['tokens'].setdefault(token_id, {})['where'] = {'confusion_probability': list(confusion_probability)}

    def what_task(self, token_id, index, phonemes, phonemes_probability):
        self['tokens'].setdefault(token_id, {}) \
            .setdefault('what', {}) \
            .setdefault('per_index_phonemes_probability', {}) \
            .setdefault(str(index), {})[phonemes] = phonemes_probability

    def full_task(self, token_id, pronunciation, pronunciation_probability):
        self['tokens'].setdefault(token_id, {}) \
            .setdefault('full', {}) \
            .setdefault('pronunciations_probability', {})[pronunciation] = pronunciation_probability

    def __repr__(self):
        return json.dumps(self, indent=2)


def load_wordlist(wordlist_filename):
    return set(common.parse_wordlist(wordlist_filename))


def load_lexicon(lexicon_filename):
    return Lexicon({k: [pron.split(' ') for pron in v]
                    for k, v in common.parse_dictionary(lexicon_filename).items()})


def load_dataset(dataset_filename):
    with open(dataset_filename) as dataset_file:
        return Dataset(json.load(dataset_file))

    return {}


def save_submission(submission, submission_filename):
    with open(submission_filename, 'w') as f:
        json.dump(submission, f, indent=2)

    return


def _combine_dicts(*args):
    all_dict = []
    for arg in args:
        all_dict += list(arg.items())

    return dict(all_dict)


def get_edit_scripts(pron_a, pron_b):
    """Get the edit scripts to transform between two given pronunciations.

    :param pron_a: Source pronunciation as list of strings, each string corresponding to a phoneme
    :param pron_b: Target pronunciation as list of strings, each string corresponding to a phoneme
    :return: List of edit scripts.  Each edit script is represented as a list of operations,
                where each operation is a dictionary.
    """
    distance, scripts, costs, ops = edit_distance.best_transforms(pron_a, pron_b)
    return [script.to_primitive() for script in scripts]


def print_edit_script(edit_script):
    """Print an edit script to the terminal.

    :param edit_script: The edit script as a list of operations, where each operation is a dictionary.
    """
    print('{}\n{}'.format(*edit_script_to_strings(edit_script)))


def edit_script_to_strings(edit_script, use_colors=True):
    """Convert an edit script to a pair of strings representing the operation in a human readable way.

    :param edit_script: The edit script as a list of operations, where each operation is a dictionary.
    :param use_colors: Boolean indicating whether to use terminal color codes to color the output.
    :return: Tuple with text corresponding to the first pronunciation and the text of the second one.
    """
    colors = collections.defaultdict(str)

    if use_colors:
        colors['red'] = '\x1b[31m'
        colors['normal'] = '\x1b[m'
        colors['green'] = '\x1b[32m'
        colors['on_red'] = '\x1b[41m'

    src_txt = ''
    dst_txt = ''
    for op in edit_script:
        if op['op_code'] == 'match':
            width = max(len(op['from_symbol']), len(op['to_symbol']))
            if op['from_symbol'] == op['to_symbol']:
                src_txt += '{green}{from_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                            op,
                                                                                            {'width': width}))
                dst_txt += '{green}{to_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                          op,
                                                                                          {'width': width}))
            else:
                src_txt += '{red}{from_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                          op,
                                                                                          {'width': width}))
                dst_txt += '{red}{to_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                        op,
                                                                                        {'width': width}))

        elif op['op_code'] == 'insert':
            space = ' '*len(op['to_symbol'])
            src_txt += '{on_red}{space}{normal}'.format(space=space, **_combine_dicts(colors,  op))
            dst_txt += '{red}{to_symbol}{normal}'.format(**_combine_dicts(colors, op))

        elif op['op_code'] == 'delete':
            space = ' '*len(op['from_symbol'])
            src_txt += '{red}{from_symbol}{normal}'.format(**_combine_dicts(colors, op))
            dst_txt += '{on_red}{space}{normal}'.format(space=space, **_combine_dicts(colors, op))

        src_txt += ' '
        dst_txt += ' '

    return src_txt, dst_txt

