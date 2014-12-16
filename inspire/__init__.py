#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspire.py
Collection of functions useful for the INSPIRE challenge.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy

import json
import logging
import os
import pprint as pp
import collections
import re
import shlex
import urllib2
import progressbar as pb

from . import common
from . import edit_distance
import time

UTF8_NORMALIZATION = 'NFD'

BASE_URL = 'http://46.226.110.12:5000'


def _get_url(url):
    try:
        u = urllib2.urlopen(url)
        resp = u.read()
        u.close()
    except urllib2.HTTPError:
        logging.error('Could not get: \n'
                      '{}\n'
                      'Please check your internet connection.\n'
                      'If problem persists inform the challenge organizers.'.format(url))
        return '{}'

    return resp


def _download_url(url, filename=None):
    try:
        u = urllib2.urlopen(url)
    except urllib2.HTTPError:
        logging.error('Could not download: \n'
                      '{}\n'
                      'Please check your internet connection.\n'
                      'If problem persists inform the challenge organizers.'.format(url))
        return None

    meta = u.info()

    reported_filename = shlex.split(re.findall(r'filename=(\S+)',
                                               meta.getheaders('Content-Disposition')[0])[0])[0]

    filename = filename or reported_filename

    if os.path.isfile(filename):
        u.close()
        return filename

    f = open(filename, 'wb')
    file_size = int(meta.getheaders("Content-Length")[0])

    widgets = ['Downloading: ', pb.Percentage(), ' ', pb.Bar(),
               ' ', pb.ETA(), ' ', pb.FileTransferSpeed()]

    pbar = pb.ProgressBar(widgets=widgets, maxval=file_size).start()

    file_size_dl = 0
    block_sz = 8192
    while True:
        download_buffer = u.read(block_sz)
        if not download_buffer:
            break

        file_size_dl += len(download_buffer)
        f.write(download_buffer)

        pbar.update(file_size_dl)

    pbar.finish()

    print('Finished downloading {}'.format(filename))

    f.close()
    u.close()
    return filename


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


class EvaluationSetting(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        return json.dumps(self, indent=2)

    def download_dataset(self, filename=None):
        return _download_url('{}/download/dataset/{}'.format(BASE_URL, self['id']), filename=filename)

    def download_lexicon(self, filename=None):
        return _download_url('{}/download/lexicon/{}'.format(BASE_URL, self['id']), filename=filename)


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
    def __init__(self, email='', evaluation_setting=None, description=''):
        self.evaluation_setting = evaluation_setting

        if self.evaluation_setting is None:
            raise ValueError('Must set the evaluation_setting, when constructing a Submission.')

        evaluation_setting_id = self.evaluation_setting['id'] if self.evaluation_setting is not None else None

        self.update({'metadata': {'email': email,
                                  'description': description,
                                  'evaluation_setting_id': evaluation_setting_id},
                     'tokens': {}})

    def where_task(self, token_id, confusion_probability):
        self['tokens'].setdefault(token_id, {})['where'] = list(confusion_probability)

    def what_task(self, token_id, index, phonemes, phonemes_probability):
        self['tokens'].setdefault(token_id, {}) \
            .setdefault('what', {}) \
            .setdefault(str(index), {})[phonemes] = phonemes_probability

    def full_task(self, token_id, pronunciation, pronunciation_probability):
        """Provide the prediction of the full task.

        This is function is used to predict the porbability of a given pronunciation being reported for a given token.

        :param token_id: The token for which the prediction is being provided
        :param pronunciation: The pronunciation for which the prediction is being provided
        :param pronunciation_probability: The probability of the pronunciation for the given token
        """

        if not 0. < pronunciation_probability < 1.:
            logging.warning()

        self['tokens'].setdefault(token_id, {}) \
            .setdefault('full', {})[pronunciation] = pronunciation_probability

    def save(self, filename):
        """Save the submission into a file.

        :param filename: where to save the submission in JSON format
        """
        with open(filename, 'w') as f:
            json.dump(self, f, indent=2)

    def submit(self, password=''):
        """Submits the participation to the web site.

        The passwords is sent as plain text.

        :return: the evaluation results.
        """

        data = json.dumps({'email': self['metadata']['email'],
                           'password': password,
                           'submission': self})

        req = urllib2.Request('{}/api/submit_with_login'.format(BASE_URL),
                              data, {'Content-Type': 'application/json'})
        f = urllib2.urlopen(req)
        resp = f.read()
        f.close()

        try:
            response = json.loads(resp)

        except urllib2.HTTPError as e:
            logging.error('Error while submitting the participation. {}'.format(e))
            return Job()

        if 'error' in response:
            logging.error('Error while processing the participation. {}'.format(response['error']))
            return Job()

        return Job(response)

    def evaluate(self, password=''):
        """Evaluates the development set.

        The passwords is sent as plain text.

        :return: the evaluation results.
        """

        # Make a copy only keeping the development set
        dev_submission = copy.deepcopy(self)
        dev_submission['tokens'] = {token_id: token for token_id, token in self['tokens'].items()
                                    if token_id in self.evaluation_setting['development_set']}

        data = json.dumps({'email': dev_submission['metadata']['email'],
                           'password': password,
                           'submission': dev_submission})

        req = urllib2.Request('{}/api/evaluate_with_login'.format(BASE_URL),
                              data, {'Content-Type': 'application/json'})
        f = urllib2.urlopen(req)
        resp = f.read()
        f.close()

        try:
            response = json.loads(resp)

        except urllib2.HTTPError as e:
            logging.error('Error while submitting the participation. {}'.format(e))
            return Job()

        if 'error' in response:
            logging.error('Error while processing the participation. {}'.format(response['error']))
            return Job()

        return Job(response)

    def __repr__(self):
        return json.dumps(self, indent=2)


class Job(dict):
    _ready_states = {'SUCCESS', 'FAILURE'}

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        return json.dumps(self, indent=2)

    def status(self):
        if 'task_id' not in self:
            logging.error('This job has not succeed.')
            return {}

        req = urllib2.Request('{}/api/task/{}/status'.format(BASE_URL, self['task_id']))

        f = urllib2.urlopen(req)
        resp = f.read()
        f.close()

        return json.loads(resp)

    def wait(self):
        widgets = ['Processing: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]

        pbar = pb.ProgressBar(widgets=widgets, maxval=100).start()

        while True:
            task_status = self.status()['task']

            if task_status['status'] == 'PROGRESS':
                if 'count' in task_status['result'] and 'total' in task_status['result']:
                    progress = int(float(task_status['result']['count']) / task_status['result']['total'] * 100)
                else:
                    progress = None

                pbar.update(progress)

            elif task_status['status'] in self._ready_states:
                pbar.finish()
                break

            else:
                pbar.update()

            time.sleep(1.0)

        return

    def result(self):
        if 'task_id' not in self:
            logging.error('This job has not succeed.')
            return {}

        req = urllib2.Request('{}/api/task/{}/result'.format(BASE_URL, self['task_id']))

        f = urllib2.urlopen(req)
        resp = f.read()
        f.close()

        data = json.loads(resp)

        return data['task']['result']

    def ready(self):
        return self.status()['task']['status'] in self._ready_states


def get_evaluation_setting(setting_id=None):
    evaluation_setting_url = '{}/api/evaluation_setting'.format(BASE_URL)
    if setting_id is not None:
        evaluation_setting_url += '/{}'.format(setting_id)

    return EvaluationSetting(json.loads(_get_url(evaluation_setting_url)))


def load_wordlist(wordlist_filename):
    return set(common.parse_wordlist(wordlist_filename))


def load_lexicon(lexicon_filename):
    return Lexicon({k: [pron.split(' ') for pron in v]
                    for k, v in common.parse_dictionary(lexicon_filename).items()})


def loads_lexicon(lexicon_file):
    return Lexicon({k: [pron.split(' ') for pron in v]
                    for k, v in common.parses_dictionary(lexicon_file).items()})


def load_dataset(dataset_filename):
    with open(dataset_filename) as dataset_file:
        return loads_dataset(dataset_file)

    return {}


def loads_dataset(dataset_file):
    return Dataset(json.load(dataset_file))


def _combine_dicts(*args):
    all_dict = []
    for arg in args:
        all_dict += list(arg.items())

    return dict(all_dict)


def get_edit_scripts(pron_a, pron_b, edit_costs=(1.0, 1.0, 1.0)):
    """Get the edit scripts to transform between two given pronunciations.

    :param pron_a: Source pronunciation as list of strings, each string corresponding to a phoneme
    :param pron_b: Target pronunciation as list of strings, each string corresponding to a phoneme
    :param edit_costs: Costs of insert, replace and delete respectively
    :return: List of edit scripts.  Each edit script is represented as a list of operations,
                where each operation is a dictionary.
    """
    op_costs = {'insert': lambda x: edit_costs[0],
                'match': lambda x, y: 0 if x == y else edit_costs[1],
                'delete': lambda x: edit_costs[2]}

    distance, scripts, costs, ops = edit_distance.best_transforms(pron_a, pron_b, op_costs=op_costs)

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

