#!/usr/bin/env python
"""htk
Reading and writing HTK files.

(Currently only handles diagonal covariance mixture models)
"""

import numpy as np
import re
from collections import namedtuple
import struct
import codecs
import unicodedata

from . import htk_model

Label = namedtuple('Label', 'start end symbol log_likelihood word')

ParameterMeta = namedtuple('ParameterMeta', 'n_samples samp_period samp_size parm_kind '
                                            'parm_kind_base parm_kind_opts parm_kind_str')
Parameter = namedtuple('Parameter', 'meta samples')

HTK_HYPOTHESIS_PATTERN = r'''
    (\s*
        ((?P<start>\d+)
         [ \t\f\v]?)?
        ((?P<end>\d+)?
         [ \t\f\v]?)?
        ((?P<symbol>\S+)
         [ \t\f\v]?)
        ((?P<log_likelihood>[-+]?\d+\.?\d*)?
         [ \t\f\v]?)?
        ((?P<word>\S+))?
    \s*)
    '''

HTK_MLF_PATTERN = r'''
      (?:^\"(?P<file>.*?)\"$)   # The file name
      (?P<hypotheses>.*?)
      (?:^\.$)
    '''

HTK_HYPOTHESIS_RE = re.compile(HTK_HYPOTHESIS_PATTERN, re.MULTILINE | re.VERBOSE | re.UNICODE)
HTK_MLF_RE = re.compile(HTK_MLF_PATTERN, re.MULTILINE | re.VERBOSE | re.UNICODE | re.DOTALL)

HTK_PARAMETER_KINDS = {
    0: 'WAVEFORM',  # sampled waveform
    1: 'LPC',  # linear prediction filter coefficients
    2: 'LPREFC',  # linear prediction reflection coefficients
    3: 'LPCEPSTRA',  # LPC cepstral coefficients
    4: 'LPDELCEP',  # LPC cepstra plus delta coefficients
    5: 'IREFC',  # LPC reflection coef in 16 bit integer format
    6: 'MFCC',  # mel-frequency cepstral coefficients
    7: 'FBANK',  # log mel-filter bank channel outputs
    8: 'MELSPEC',  # linear mel-filter bank channel outputs
    9: 'USER',  # user defined sample kind
    10: 'DISCRETE',  # vector quantised data
}

HTK_PARAMETER_FLAGS = [
    ('E', 0o000100),  # has energy
    ('N', 0o000200),  # absolute energy suppressed
    ('D', 0o000400),  # has delta coefficients
    ('A', 0o001000),  # has acceleration coefficients
    ('C', 0o002000),  # is compressed
    ('Z', 0o004000),  # has zero mean static coef.
    ('K', 0o010000),  # has CRC checksum
    ('O', 0o020000),  # has 0'th cepstral coef.
    ('V', 0o040000),  # has VQ data
    ('T', 0o100000),  # has third differential coef.
]


def _htk_str_to_param(parm_str):
    tokens = parm_str.split('_')

    inv_htk_parameter_kinds = dict([(v, k) for k, v in HTK_PARAMETER_KINDS.items()])
    dict_htk_parameter_flags = dict(HTK_PARAMETER_FLAGS)

    parm_kind = inv_htk_parameter_kinds[tokens[0]]

    for flag in tokens[1:]:
        parm_kind = dict_htk_parameter_flags[flag]

    return parm_kind


def _htk_param_to_str(parm_kind):
    # first 6 bits are the main parameter kind
    result = HTK_PARAMETER_KINDS[parm_kind & 0b000000111111]

    # check for the flags
    for flag, mask in HTK_PARAMETER_FLAGS:
        if parm_kind & mask:
            result += '_' + flag

    return result


def _htk_param_to_base_options(parm_kind):
    # first 6 bits are the main parameter kind
    base = HTK_PARAMETER_KINDS[parm_kind & 0b000000111111]

    # check for the flags
    options = []
    for flag, mask in HTK_PARAMETER_FLAGS:
        if parm_kind & mask:
            options.append('_' + flag)

    return base, options


HTK_HEADER_FORMAT = '>iihh'


def _parse_parameter_meta(input_file):
    buffer = input_file.read(struct.calcsize(HTK_HEADER_FORMAT))
    n_samples, samp_period, samp_size, parm_kind = struct.unpack_from(HTK_HEADER_FORMAT, buffer)
    parm_kind_str = _htk_param_to_str(parm_kind)
    parm_kind_base, parm_kind_opts = _htk_param_to_base_options(parm_kind)

    if '_C' in parm_kind_str:
        raise NotImplementedError('HTK files with compression flag (_C) not supported.')

    if '_K' in parm_kind_str:
        raise NotImplementedError('HTK files with CRC flag (_K) not supported.')

    meta = ParameterMeta(n_samples=n_samples,
                         samp_period=samp_period,
                         samp_size=samp_size,
                         parm_kind=parm_kind,
                         parm_kind_base=parm_kind_base,
                         parm_kind_opts=parm_kind_opts,
                         parm_kind_str=parm_kind_str)
    return meta


def _parse_parameter_samples(input_file, meta):
    samples = []
    sample_format = '>%df' % (meta.samp_size / 4)
    for i in range(meta.n_samples):
        buffer = input_file.read(struct.calcsize(sample_format))
        sample = struct.unpack_from(sample_format, buffer)
        samples.append(list(sample))

    return Parameter(meta=meta,
                     samples=np.array(samples))


def _parse_parameter(input_file):
    meta = _parse_parameter_meta(input_file)
    return _parse_parameter_samples(input_file, meta)


def _serialize_parameter(parameter, output_file):
    output_file.write(struct.pack(HTK_HEADER_FORMAT,
                                  parameter.meta.n_samples,
                                  parameter.meta.samp_period,
                                  parameter.meta.samp_size,
                                  parameter.meta.parm_kind))

    sample_dimension = parameter.meta.samp_size / 4
    sample_format = '>%df' % (sample_dimension,)
    for i, sample in enumerate(parameter.samples):
        if len(sample) != sample_dimension:
            raise ValueError('All the samples written to an HTK file must have the same size.'
                             'Sample {} has length {}, while previous samples had length {}'.format(i,
                                                                                                    len(sample),
                                                                                                    sample_dimension))

        output_file.write(struct.pack(sample_format, *tuple(sample)))
    return


def load_parameter(input_filename, only_header=False):
    """Load HTK parameter/feature file.

    :param input_filename:
    :param only_header: only load the metadata
    :return: a named tuple representing the HTK parmeter file
    """

    with open(input_filename, 'rb') as f:
        meta = _parse_parameter_meta(f)
        if only_header:
            return meta

        return _parse_parameter_samples(f, meta)


def save_parameter(parameter, output_filename):
    """Save a file in HTK Parameter File Format.

    :param parameter: The named tuple Parameter containing the parameter file metadata and data
    :param output_filename: The name of the file where to save
    """
    with open(output_filename, 'wb') as f:
        _serialize_parameter(parameter, f)


def create_parameter(samples, sample_period):
    """Create a HTK Parameter object from an array of samples and a samples period

    :param samples (list of lists or array of floats): The samples to write into the file. Usually feature vectors.
    :param sample_period (int): Sample period in 100ns units.
    """

    parm_kind_str = 'USER'
    parm_kind = _htk_str_to_param(parm_kind_str)
    parm_kind_base, parm_kind_opts = _htk_str_to_param(parm_kind_str)

    meta = ParameterMeta(n_samples=len(samples),
                         samp_period=sample_period,
                         samp_size=len(samples[0]) * 4,  # size in bytes
                         parm_kind_str=parm_kind_str,
                         parm_kind=parm_kind,
                         parm_kind_base=parm_kind_base,
                         parm_kind_opts=parm_kind_opts)
    return Parameter(meta=meta,
                     samples=np.array(samples))


def load_mlf(filename, utf8_normalization=None):
    """Load an HTK Master Label File.

    :param filename: The filename of the MLF file.
    :param utf8_normalization: None
    """
    with codecs.open(filename, 'r', 'string_escape') as f:
        data = f.read().decode('utf8')
        if utf8_normalization:
            data = unicodedata.normalize(utf8_normalization, data)

    mlfs = {}
    for mlf_object in HTK_MLF_RE.finditer(data):
        mlfs[mlf_object.group('file')] = [[Label(**mo.groupdict())
                                           for mo
                                           in HTK_HYPOTHESIS_RE.finditer(recognition_data)]
                                          for recognition_data
                                          in re.split(r'\n///\n', mlf_object.group('hypotheses'))]

    return mlfs


def save_mlf(mlf, output_filename):
    """Save an HTK Master Label File.

    :param mlf: MLF dictionary containing a mapping from file to list of annotations.
    :param output_filename: The file where to save the MLF
    """
    with codecs.open(output_filename, 'w', 'utf-8') as f:
        f.write(u'#!MLF!#\n')
        for k, v in mlf.items():
            f.write(u'"{}"\n'.format(k))
            for labels in v:
                for label in labels:
                    line = u'{start} {end} {symbol} ' \
                           u'{loglikelihood} {word}'.format(start=label.start or '',
                                                            end=label.end or '',
                                                            symbol=label.symbol or '',
                                                            loglikelihood=label.log_likelihood or '',
                                                            word=label.word or '')
                    f.write(u'{}\n'.format(line.strip()))

                f.write(u'.\n')


def load_model(filename):
    """Load an HTK HMM Model file.

    Parse an hmmdef file and return dictionaries of hmms, states and
    transition matrices
    """
    return htk_model.load_model(filename)


def save_model(model, filename):
    """Save an HTK HMM Model file.
    """
    htk_model.save_model(model, filename)


def main():
    """Test code called from commandline"""
    model = load_model('../data/hmmdefs')
    hmm = model.hmms['r-We']
    for state_name in hmm.state_names:
        print(state_name)
        state = model.states[state_name]
        print(state.means_)
    print(model)
    model2 = load_model('../data/prior.hmm1mixSI.rate32')
    print(model2)


if __name__ == '__main__':
    main()
