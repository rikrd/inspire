#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspire.py
Collection of functions useful for the INSPIRE challenge.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import gzip

import gzip
import json
import logging
import os
import pprint as pp
import collections
import re
import shlex
import urllib2
import io
import itertools
import requests
import requests.exceptions
import zipfile
import StringIO
import progressbar as pb
from scipy.io import wavfile
from contextlib import closing

from . import common
from . import edit_distance
import time

UTF8_NORMALIZATION = 'NFD'

BASE_URL = 'http://46.226.110.12:5000'
BASE_URL = 'http://localhost:5000'


def _load_zip_wav(zfile, offset=0, count=None):
    """Load a wav file into an array from frame start to fram end

    :param zfile: ZipExtFile file-like object from where to load the audio
    :param offset: First sample to load
    :param count: Maximum number of samples to load
    :return: The audio samples in a numpy array of floats
    """

    buf = StringIO.StringIO(zfile.read())

    sample_rate, audio = wavfile.read(buf)
    audio = audio[offset:]

    if count:
        audio = audio[:count]

    return sample_rate, audio


def _get_url(url):
    r = requests.get(url)

    if r.status_code != 200:
        logging.error('Could not get: \n'
                      '{}\n'
                      'Please check your internet connection.\n'
                      'If problem persists inform the challenge organizers.'.format(url))
        return '{}'

    return r.content


def _download_url(url, filename=None):
    with closing(requests.get(url, stream=True)) as r:
        if r.status_code != 200:
            logging.error('Could not download: \n'
                          '{}\n'
                          'Please check your internet connection.\n'
                          'If problem persists inform the challenge organizers.'.format(url))
            return None

        reported_filename = os.path.basename(url)
        if r.headers.get('Content-Disposition'):
            reported_filename = shlex.split(re.findall(r'filename=(\S+)',
                                                       r.headers['Content-Disposition'])[0])[0]

        filename = filename or reported_filename

        if os.path.isfile(filename):
            return filename

        with open(filename, 'wb') as f:
            file_size = r.headers.get('content-length')

            if file_size is None:  # no content length header
                f.write(r.content)
            else:
                file_size_dl = 0
                file_size = int(file_size)

                widgets = ['Downloading: ', pb.Percentage(), ' ', pb.Bar(),
                           ' ', pb.ETA(), ' ', pb.FileTransferSpeed()]
                pbar = pb.ProgressBar(widgets=widgets, maxval=file_size).start()
                chunk_size = 10240
                for data in r.iter_content(chunk_size=chunk_size):
                    file_size_dl = min(file_size_dl + chunk_size, file_size)
                    f.write(data)

                    pbar.update(file_size_dl)

                pbar.finish()

            print('Finished downloading {}'.format(filename))

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
    def __init__(self, email=None, evaluation_setting=None, description=None, metadata={}):
        self['metadata'] = {'email': email,
                            'evaluation_setting': evaluation_setting,
                            'description': description}

        self['metadata'].update(metadata)

        if self['metadata']['evaluation_setting'] is None:
            raise ValueError('Must set the evaluation_setting, when constructing a Submission.')

        self['tokens'] = {}

    def where_task(self, token_id, confusion_probability):
        """Provide the prediction of the where task.

        This function is used to predict the probability of a given pronunciation being reported for a given token.

        :param token_id: The token for which the prediction is being provided
        :param confusion_probability: The list or array of confusion probabilities at each index
        :param phonemes_probability: The probability of the phoneme or phoneme sequence
        """

        self['tokens'].setdefault(token_id, {})['where'] = list(confusion_probability)

    def what_task(self, token_id, index, phonemes, phonemes_probability, warn=True):
        """Provide the prediction of the what task.

        This function is used to predict the probability of a given phoneme being reported at a given index
        for a given token.

        :param token_id: The token for which the prediction is provided
        :param index: The index of the token for which the prediction is provided
        :param phonemes: The phoneme or phoneme sequence for which the prediction is being made
        (as a space separated string)
        :param phonemes_probability: The probability of the phoneme or phoneme sequence
        :param warn: Set to False in order to avoid warnings about 0 or 1 probabilities
        """

        if not 0. < phonemes_probability < 1. and warn:
            logging.warning('Setting a probability of [{}] to phonemes [{}] for token [{}].\n '
                            'Using probabilities of 0.0 or 1.0 '
                            'may lead to likelihoods of -Infinity'.format(phonemes_probability,
                                                                          phonemes,
                                                                          token_id))

        self['tokens'].setdefault(token_id, {}) \
            .setdefault('what', {}) \
            .setdefault(str(index), {})[phonemes] = phonemes_probability

    def full_task(self, token_id, pronunciation, pronunciation_probability, warn=True):
        """Provide the prediction of the full task.

        This function is used to predict the probability of a given pronunciation being reported for a given token.

        :param token_id: The token for which the prediction is provided
        :param pronunciation: The pronunciation for which the prediction is being made (as a list of strings
        or space separated string)
        :param pronunciation_probability: The probability of the pronunciation for the given token
        :param warn: Set to False in order to avoid warnings about 0 or 1 probabilities
        """

        if not 0. < pronunciation_probability < 1. and warn:
            logging.warning('Setting a probability of [{}] to pronunciation [{}] for token [{}].\n '
                            'Using probabilities of 0.0 or 1.0 '
                            'may lead to likelihoods of -Infinity'.format(pronunciation_probability,
                                                                          pronunciation,
                                                                          token_id))

        key = pronunciation
        if isinstance(key, list):
            if not all([isinstance(phoneme, basestring) for phoneme in key]):
                raise ValueError('The pronunciation must be of type string (a sequence of space separated phonemes) '
                                 'or of type list (containing phonemes of type strings).'
                                 'User supplied: {}'.format(key))

            key = ' '.join(pronunciation)

        self['tokens'].setdefault(token_id, {}) \
            .setdefault('full', {})[key] = pronunciation_probability

    def dumps(self):
        bytes = io.BytesIO()
        self.dump(bytes)
        return bytes.getvalue()

    def dump(self, fileobj):
        with gzip.GzipFile(fileobj=fileobj, mode='w') as z:
            metadata = copy.copy(self['metadata'])
            metadata['token_count'] = len(self['tokens'])

            z.write(json.dumps(metadata))
            z.write('\n')

            for token_id, token in self['tokens'].items():
                z.write(json.dumps((token_id, token), sort_keys=True))
                z.write('\n')

    def save(self, filename):
        """Save the submission into a file.

        :param filename: where to save the submission
        """
        with open(filename, 'w') as f:
            self.dump(f)

        return

    # TODO: use a limit when ready http://bugs.python.org/issue15955

    @staticmethod
    def open(filename):
        """Open the submission from a file.

        :param filename: where to load the submission from
        """
        with open(filename, 'r') as f:
            return self.load(f)

    @staticmethod
    def open_metadata(filename):
        """Open the submission from a file.

        :param filename: where to load the submission from
        """
        with open(filename, 'r') as f:
            return self.load_metadata(f)

    @staticmethod
    def open_tokens(filename):
        """Open the submission from a file.

        :param filename: where to load the submission from
        """
        with open(filename, 'r') as f:
            return self.load_tokens(f)

    @staticmethod
    def loads(data):
        fileobj = io.BytesIO(data)
        return Submission.load(fileobj)

    @staticmethod
    def load(fileobj):
        """Load the submission from a file-like object

        :param fileobj: File-like object
        :return: the loaded submission
        """
        with gzip.GzipFile(fileobj=fileobj, mode='r') as z:
            submission = Submission(metadata=json.loads(z.readline()))

            for line in z:
                token_id, token = json.loads(line)
                submission['tokens'][token_id] = token

        return submission

    @staticmethod
    def loads_metadata(data):
        fileobj = io.BytesIO(data)
        return Submission.load_metadata(fileobj)

    @staticmethod
    def load_metadata(fileobj):
        """Load the submission from a file.

        :param filename: where to load the submission from
        """
        with gzip.GzipFile(fileobj=fileobj, mode='r') as z:
            return json.loads(z.readline())

    @staticmethod
    def loads_tokens(data):
        fileobj = io.BytesIO(data)
        return Submission.load_tokens(fileobj)

    @staticmethod
    def load_tokens(fileobj):
        with gzip.GzipFile(fileobj=fileobj, mode='r') as z:
            z.readline()  # skip the metadata

            for line in z:
                token = json.loads(line)

                yield token

    def submit(self, password=''):
        """Submits the participation to the web site.

        The passwords is sent as plain text.

        :return: the evaluation results.
        """

        url = '{}/api/submit'.format(BASE_URL)
        try:
            r = requests.post(url,
                              data=self.dumps(),
                              headers={'content-type': 'application/json'},
                              auth=(self['metadata']['email'], password))

            response = r.json()

        except requests.exceptions.HTTPError as e:
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
        dev_submission = self
        if self['metadata'].get('evaluation_setting', {}).get('development_set', None):
            dev_submission = copy.deepcopy(self)
            dev_submission['tokens'] = {token_id: token for token_id, token in self['tokens'].items()
                                        if token_id in self['metadata']['evaluation_setting']['development_set']}

        url = '{}/api/evaluate'.format(BASE_URL)
        try:
            r = requests.post(url,
                              data=dev_submission.dumps(),
                              headers={'content-type': 'application/json'},
                              auth=(dev_submission['metadata']['email'], password))

            response = r.json()

        except requests.exceptions.HTTPError as e:
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

        url = '{}/api/task/{}/status'.format(BASE_URL, self['task_id'])
        r = requests.get(url)

        return r.json()

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

        url = '{}/api/task/{}/result'.format(BASE_URL, self['task_id'])
        r = requests.get(url)

        data = r.json()

        return data['task']['result']

    def ready(self):
        return self.status()['task']['status'] in self._ready_states


def download_dataset_audio(filename='dataset_audio.zip'):
    return _download_url('{}/static/audio.zip'.format(BASE_URL), filename=filename)


def get_token_audio(token_id, dataset_audio_filename, dataset):
    token = dataset['tokens'][token_id]

    with zipfile.ZipFile(dataset_audio_filename) as zf:
        signal_name = 'confusionWavs/wavs/{}'.format(token['signal_wav'])

        with zf.open(signal_name) as f:
            sample_rate, signal_audio = _load_zip_wav(f)

    speech_audio = signal_audio[:, 0]
    noise_audio = signal_audio[:, 1]

    return sample_rate, speech_audio, noise_audio


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

    return [full_edit_script(script.to_primitive()) for script in scripts]


def full_edit_script(script):
    previous_index = None
    last_index = -1

    new_script = []

    for op in script:
        if op['index'] != previous_index:
            new_op = copy.copy(op)

            # Convert None to ''
            new_op['from_symbol'] = new_op['from_symbol'] or ''
            new_op['to_symbol'] = new_op['to_symbol'] or ''

            previous_index = op['index']

            # Fill gaps
            while last_index < op['index']-1:
                new_script.append({'index': last_index + 1,
                                   'from_symbol': '',
                                   'to_symbol': '',
                                   'op_code': 'noninsert'})

                last_index = new_script[-1]['index']

            new_script.append(new_op)
            last_index = new_script[-1]['index']

        else:
            # Merge consecutive inserted symbols
            if new_script[-1]['to_symbol']:
                new_script[-1]['to_symbol'] += ' {}'.format(op['to_symbol'])

            # Merge consecutive deleted symbols
            # (this doesn't happen since they have different indices)
            if new_script[-1]['from_symbol']:
                new_script[-1]['from_symbol'] += ' {}'.format(op['from_symbol'])

    if last_index % 2 == 1:
        # Missing the last gap
        new_script.append({'index': last_index + 1,
                           'from_symbol': '',
                           'to_symbol': '',
                           'op_code': 'noninsert'})

    return new_script


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

        elif op['op_code'] == 'noninsert':
            continue

        src_txt += ' '
        dst_txt += ' '

    return src_txt, dst_txt

