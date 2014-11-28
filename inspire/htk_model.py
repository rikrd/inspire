#!/usr/bin/env python
"""load_htk_model
Reads an HTK model file

Usage:
    load_htk_model <model_file>... [options]
    load_htk_model --help

Options:
    <model_file>                                    the files of the model (e.g. macros hmmdefs)
    -o <output_file>, --output_file <output_file>   the output file (default is stdout)
    -h, --help                                      print this help screen
"""

from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'rmarxer'

import docopt
import numpy as np
import json
import collections

from . import htk_model_parser


def _to_ordered_dict(ast):
    result = collections.OrderedDict()
    for k, v in ast.items():
        result[k] = v

    return result


class HtkModelSemantics(object):
    def __init__(self):
        pass

    def matrix(self, ast):
        #return [float(v) for v in ast.split(' ')]
        return np.fromstring(ast, sep=' ')

    def vector(self, ast):
        #return [float(v) for v in ast.split(' ')]
        return np.fromstring(ast, sep=' ')

    def short(self, ast):
        return int(ast)

    def float(self, ast):
        return float(ast)

    def transPdef(self, ast):
        d = _to_ordered_dict(ast)
        d['matrix'] = d['array'].reshape((ast['dim'], ast['dim']))
        d.pop('array')
        return d

    def _default(self, ast):
        if isinstance(ast, collections.Mapping):
            return _to_ordered_dict(ast)

        return ast

    def _unquote(self, txt):
        if txt.startswith('"') and txt.endswith('"'):
            return txt[1:-1]

        return txt

    def string(self, ast):
        return self._unquote(ast)

    def __repr__(self):
        return ''


class NumPyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


def load_model(*args):
    """Load an HTK model from one ore more files.

    :param args: Filenames of the model (e.g. macros hmmdefs)
    :return: The model as an OrderedDict()
    """
    text = ''
    for fnm in args:
        text += open(fnm).read()
        text += '\n'

    parser = htk_model_parser.htk_modelParser()
    model = HtkModelSemantics()
    return parser.parse(text,
                        rule_name='model',
                        ignorecase=True,
                        semantics=model,
                        comments_re="\(\*.*?\*\)",
                        trace=False)


def _array_to_htk(arr):
    result = ''

    for row in arr:
        result += ' {}\n'.format(' '.join(['{:2.6e}'.format(value) for value in row]))

    return result


def _serialize_option(option):
    result = ''
    if option.get('hmm_set_id', None) is not None:
        result += '<HmmSetId> {}'.format(option['hmm_set_id'])

    if option.get('stream_info', None) is not None:
        result += '<StreamInfo> {}'.format(option['stream_info']['count'])

        if option['stream_info'].get('sizes', None) is not None:
            result += ' {}'.format(' '.join(['{:d}'.format(v) for v in option['stream_info']['sizes']]))

    if option.get('vector_size', None) is not None:
        result += '<VecSize> {:d}'.format(option['vector_size'])

    if option.get('input_transform', None) is not None:
        raise NotImplementedError('Serialization of {} '
                                  'is not implemented.'.format(option['input_transform']))

    if option.get('covariance_kind', None) is not None:
        result += '<{}>'.format(option['covariance_kind'])

    if option.get('duration_kind', None) is not None:
        result += '<{}>'.format(option['duration_kind'])

    if option.get('parameter_kind', None) is not None:
        result += '<{}{}>'.format(option['parameter_kind']['base'],
                                  ''.join(option['parameter_kind']['options']))

    result += '\n'
    return result


def _serialize_transp(definition):
    if isinstance(definition, basestring):
        return '~t "{}"\n'.format(definition)

    result = ''
    result += '<TransP> {}\n'.format(definition['dim'])
    result += '{}'.format(_array_to_htk(definition['matrix']))
    return result


def _serialize_variance(definition):
    if isinstance(definition, basestring):
        return '~v {}\n'.format(definition)

    result = ''
    result += '<Variance> {}\n'.format(definition['dim'])
    result += '{}'.format(_array_to_htk(definition['vector'][None]))
    return result


def _serialize_mean(definition):
    if isinstance(definition, basestring):
        return '~u "{}"\n'.format(definition)

    result = ''
    result += '<Mean> {}\n'.format(definition['dim'])
    result += '{}'.format(_array_to_htk(definition['vector'][None]))
    return result


def _serialize_duration(definition):
    if isinstance(definition, basestring):
        return '~d "{}"\n'.format(definition)

    result = ''
    result += '<Duration> {}\n'.format(definition['dim'])
    result += '{}'.format(_array_to_htk(definition['vector'][None]))
    return result


def _serialize_weights(definition):
    if isinstance(definition, basestring):
        return '~w "{}"\n'.format(definition)

    result = ''
    result += '<SWeights> {}\n'.format(definition['dim'])
    result += '{}'.format(_array_to_htk(definition['vector'][None]))
    return result


def _serialize_covariance(definition):
    result = ''
    if definition['variance'] is not None:
        result += _serialize_variance(definition['variance'])

    else:
        raise NotImplementedError('Cannot serialize {}'.format(definition))

    return result


def _serialize_mixpdf(definition):
    if isinstance(definition, basestring):
        return '~m "{}"\n'.format(definition)

    result = ''
    if definition.get('regression_class', None) is not None:
        result += '<RClass> {}\n'.format(definition['regression_class'])

    result += _serialize_mean(definition['mean'])
    result += _serialize_covariance(definition['covariance'])

    if definition.get('gconst', None) is not None:
        result += '<GConst> {:.6e}\n'.format(definition['gconst'])

    return result


def _serialize_mixture(definition):
    result = ''

    if definition.get('index', None) is not None:
        result += '<Mixture> {} {:.6e}\n'.format(definition['index'], definition['weight'])

    result += _serialize_mixpdf(definition['pdf'])
    return result


def _serialize_stream(definition):
    result = ''

    if definition.get('dim', None) is not None:
        result += '<Stream> {}\n'.format(definition['dim'])

    if definition.get('mixtures', None) is not None:
        for mixture in definition['mixtures']:
            result += _serialize_mixture(mixture)

    else:
        raise NotImplementedError('Cannot serialize {}'.format(definition))

    return result


def _serialize_stateinfo(definition):
    if isinstance(definition, basestring):
        return '~s "{}"\n'.format(definition)

    result = ''
    if definition.get('streams_mixcount', None):
        result += '<NumMixes> {}\n'.format(' '.join(['{}'.format(v) for v in definition['streams_mixcount']]))

    if definition.get('weights', None) is not None:
        result += _serialize_weights(definition['weights'])

    for stream in definition['streams']:
        result += _serialize_stream(stream)

    if definition.get('duration', None) is not None:
        result += _serialize_duration(definition['duration'])

    return result


def _serialize_state(definition):
    result = ''

    result += '<State> {}\n'.format(definition['index'])
    result += _serialize_stateinfo(definition['state'])

    return result


def _serialize_hmm(definition):
    result = ''

    result += '<BeginHMM>\n'
    if definition.get('options', None):
        for option in definition['options']:
            result += _serialize_option(option)

    result += '<NumStates> {}\n'.format(definition['state_count'])

    for state in definition['states']:
        result += _serialize_state(state)

    if definition.get('regression_tree', None) is not None:
        raise NotImplementedError('Cannot serialize {}'.format(definition['regression_tree']))

    result += _serialize_transp(definition['transition'])

    if definition.get('duration', None) is not None:
        result += _serialize_duration(definition['duration'])

    result += '<EndHMM>\n'

    return result


def save_model(model, filename):
    """Save the model into a file.

    :param model: HTK model to be saved
    :param filename: File where to save the model
    """
    with open(filename, 'w') as f:
        f.write(serialize_model(model))


def serialize_model(model):
    """Serialize the HTK model into a file.

    :param model: Model to be serialized
    """

    result = ''

    # First serialize the macros
    for macro in model['macros']:
        if macro.get('options', None):
            result += '~o '
            for option in macro['options']['definition']:
                result += _serialize_option(option)

        elif macro.get('transition', None):
            result += '~t "{}"\n'.format(macro['transition']['name'])
            result += _serialize_transp(macro['transition']['definition'])

        elif macro.get('variance', None):
            result += '~v "{}"\n'.format(macro['variance']['name'])
            result += _serialize_variance(macro['variance']['definition'])

        elif macro.get('state', None):
            result += '~s "{}"\n'.format(macro['state']['name'])
            result += _serialize_stateinfo(macro['state']['definition'])

        elif macro.get('mean', None):
            result += '~u "{}"\n'.format(macro['mean']['name'])
            result += _serialize_mean(macro['mean']['definition'])

        elif macro.get('duration', None):
            result += '~d "{}"\n'.format(macro['duration']['name'])
            result += _serialize_duration(macro['duration']['definition'])

        else:
            raise NotImplementedError('Cannot serialize {}'.format(macro))

    for hmm in model['hmms']:
        if hmm.get('name', None) is not None:
            result += '~h "{}"\n'.format(hmm['name'])

        result += _serialize_hmm(hmm['definition'])

    return result


def model_to_json(model, **kwargs):
    """Convert a model to a JSON string.

    :param model: The OrderedDict representing the model
    :param kwargs:  Keyword arguments to be forwarded to the json.dumps method
    :return: The JSON string
    """
    return json.dumps(model, cls=NumPyArrayEncoder, **kwargs)


def main():
    arguments = docopt.docopt(__doc__)
    model = load_model(*arguments['<model_file>'])

    result = model_to_json(model, indent=2)

    if arguments['--output_file'] is not None:
        with open(arguments['--output_file'], 'w') as f:
            f.write(result)
    else:
        print(result)

    return

if __name__ == '__main__':
    main()