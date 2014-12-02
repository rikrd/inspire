#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for creating an ASR model
import os
import itertools
import shlex
import sys
import collections
import logging
import random
import tempfile
import json
import shutil
import unicodedata
import codecs
import re
import fnmatch
import copy
import math
import numpy as np

from . import config
from . import htk
from . import htk_model_utils

UTF8_NORMALIZATION = 'NFD'


def shlex_split(txt):
    return map(lambda s: s.decode('utf-8'), shlex.split(txt.encode('utf-8')))


def htk_to_utf8normalize(filename):
    with codecs.open(filename, 'r', 'string_escape') as f:
        data = unicodedata.normalize(UTF8_NORMALIZATION, f.read().decode('utf8'))

    with codecs.open(filename, 'w', 'utf-8') as f:
        f.write(data)

    return


def recursive_glob(path, pattern):
    return [os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(path)
            for f in fnmatch.filter(files, pattern)]


def parse_wordlist(list_words_filename):
    with codecs.open(list_words_filename, 'r', 'utf-8') as list_words_file:
        return [unicodedata.normalize(UTF8_NORMALIZATION, l).strip().upper() for l in list_words_file.readlines()]


def parse_dictionary(dictionary_filename):
    with codecs.open(dictionary_filename, 'r', 'utf-8') as f:
        return parses_dictionary(f.read())


def parses_dictionary(dictionary_buffer):
    dictionary = collections.defaultdict(set)

    for row, line in enumerate(dictionary_buffer.splitlines()):
        line = line.strip()
        if line.startswith('#'):
            continue

        try:
            key, value = line.split(None, 1)

        except ValueError:
            raise ValueError(u'Error parsing line {}.  "{}"'.format(row, line))

        dictionary[unicodedata.normalize(UTF8_NORMALIZATION, unicode(key))].add(
            unicodedata.normalize(UTF8_NORMALIZATION, unicode(value)))

    return dictionary


def check_words_in_dictionary(list_recordings, dictionary_filename, proposed_dictionary_filename=None):
    dictionary = parse_dictionary(dictionary_filename)
    dictionary_words = set(dictionary.keys())

    missing_words = set()
    proposed_pronunciations = collections.defaultdict(set)
    for recording in list_recordings:
        words = [w.upper() for w in get_words_from_recording(recording)]

        # if all words are in the dictionary we continue
        missing_recording_words = set(words) - dictionary_words
        if not missing_recording_words:
            continue

        missing_words |= missing_recording_words

        # otherwise we find their transcription
        proposed_pronunciations_recording = find_pronunciations(recording, missing_recording_words, dictionary)

        for key, value in proposed_pronunciations_recording.items():
            proposed_pronunciations[key] |= value

    if proposed_dictionary_filename:
        with codecs.open(proposed_dictionary_filename, 'w', 'utf-8') as f:
            for word, pronunciations in sorted(proposed_pronunciations.items()):
                for pronunciation in pronunciations:
                    f.write('{} {}\n'.format(word, pronunciation))

    return proposed_pronunciations


def find_pronunciations(recording, missing_words):
    proposed_dictionary = collections.defaultdict(set)

    words, phones = get_timit_from_recording(recording)
    if words is None or phones is None:
        return proposed_dictionary

    for missing_word in missing_words:
        # find the intervals where the missing_word appears
        intervals = [(int(v[0]), int(v[1])) for v in words if v[2].upper() == missing_word]

        for interval in intervals:
            # find the phonemes in the given interval
            phonemes = [v[2] for v in phones if int(v[0]) >= interval[0] and int(v[1]) <= interval[1]]
            pronunciation = ' '.join(phonemes)
            if pronunciation.startswith('sil '):
                pronunciation = pronunciation[len('sil '):]

            if pronunciation.endswith(' sil'):
                pronunciation = pronunciation[:-len(' sil')]

            if pronunciation:
                proposed_dictionary[missing_word].add(pronunciation)

    return proposed_dictionary


def get_speaker_from_inspire(fnm):
    return os.path.splitext(os.path.basename(fnm))[0].rsplit('_s', 1)[-1]


def get_speaker_from_wsjcam(fnm):
    speaker_filename = os.path.splitext(fnm)[0][:-2] + '00.ifo'
    if not os.path.isfile(speaker_filename):
        return None

    return os.path.split(os.path.dirname(fnm))[-1]


def get_timit_from_wsjcam(fnm):
    transcription_filename = os.path.splitext(fnm)[0] + '.wrd'
    if not os.path.isfile(transcription_filename):
        return None, None

    words = [unicodedata.normalize(UTF8_NORMALIZATION, line.strip().split()).upper() for line in
             codecs.open(transcription_filename, 'r', 'utf-8')]

    transcription_filename = os.path.splitext(fnm)[0] + '.phn'
    if not os.path.isfile(transcription_filename):
        return None, None

    phones = [line.strip().split() for line in codecs.open(transcription_filename, 'r', 'utf-8')]

    return words, phones


def get_words_from_wsjcam(fnm):
    transcription_filename = os.path.splitext(fnm)[0] + '.wrd'
    if not os.path.isfile(transcription_filename):
        return None

    words = [unicodedata.normalize(UTF8_NORMALIZATION, line.strip().split()[-1]).upper() for line in
             codecs.open(transcription_filename, 'r', 'utf-8')]

    return words


def get_words_from_inspire(fnm):
    # Always normalize when dealing with a filename (Mac +HFS uses NFD while all others use NFC normalisation)
    # Dictionaries are considered NFC
    return [unicodedata.normalize(UTF8_NORMALIZATION,
                                  os.path.splitext(os.path.basename(fnm))[0].rsplit('_s', 1)[0]).upper()]


def get_speaker_from_recording(fnm):
    return get_speaker_from_wsjcam(fnm) or get_speaker_from_inspire(fnm)


def get_words_from_recording(fnm):
    return get_words_from_wsjcam(fnm) or get_words_from_inspire(fnm)


def get_timit_from_recording(fnm):
    return get_timit_from_wsjcam(fnm)


def create_recordings_dictionary(list_selected, pronunciation_dictionary_filename, out_dictionary_filename, htk_trace,
                                 additional_dictionary_filenames=[]):
    """Create a pronunciation dictionary specific to a list of recordings"""
    temp_words_fd, temp_words_file = tempfile.mkstemp()

    words = set()
    for recording in list_selected:
        words |= set([w.upper() for w in get_words_from_recording(recording)])

    with codecs.open(temp_words_file, 'w', 'utf-8') as f:
        f.writelines([u'{}\n'.format(w) for w in sorted(list(words))])

    prepare_dictionary(temp_words_file,
                       pronunciation_dictionary_filename,
                       out_dictionary_filename,
                       htk_trace,
                       global_script_filename=config.project_path('etc/global.ded'),
                       additional_dictionary_filenames=additional_dictionary_filenames)

    os.remove(temp_words_file)

    return


def load_train_arguments(train_arguments_filename, args):
    # Set the non-supplied arguments using the training arguments
    with codecs.open(train_arguments_filename, 'r', 'utf-8') as train_arguments_file:
        train_args = json.load(train_arguments_file)

    if args.list_words_filename is None:
        args.list_words_filename = train_args['list_words_filename']

    if args.pronunciation_dictionary_filename is None:
        args.pronunciation_dictionary_filename = train_args['pronunciation_dictionary_filename']

    if args.feature_extractor_command is None:
        args.feature_extractor_command = train_args['feature_extractor_command']

    if args.feature_extractor_parameters is None:
        args.feature_extractor_parameters = train_args['feature_extractor_parameters']

    if not args.audio_dir:
        args.audio_dir = train_args['audio_dir']

    return args


def utf8_normalize(input_filename, comment_char='#', to_upper=False):
    """Normalize UTF-8 characters of a file
    """
    # Prepare the input dictionary file in UTF-8 and NFC
    temp_dict_fd, output_filename = tempfile.mkstemp()

    logging.debug('to_nfc from file {} to file {}'.format(input_filename, output_filename))

    with codecs.open(output_filename, 'w', 'utf-8') as f:
        with codecs.open(input_filename, 'r', 'utf-8') as input_f:
            lines = sorted([unicodedata.normalize(UTF8_NORMALIZATION, l)
                            for l in input_f.readlines()
                            if not l.strip().startswith(comment_char)])
            if to_upper:
                lines = [l.upper() for l in lines]

            f.writelines(lines)

    return output_filename


def merge_dictionaries(output_dictionary_filename, input_dictionary_filenames, htk_trace=0, global_script_filename=None,
                       add_sent_silence=True):
    if add_sent_silence:
        input_dictionary_filenames.append(config.project_path('etc', 'startendsilence.dict'))

    input_dictionary_in_nfc_filenames = []
    for filename in input_dictionary_filenames:
        output_filename = utf8_normalize(filename, to_upper=False)
        input_dictionary_in_nfc_filenames.append(output_filename)

    global_script = ""
    if global_script_filename:
        global_script = u"-g {}".format(global_script_filename)

    log_filename = config.temp_path('hdman.log')
    config.htk_command("HDMan -A -l {log_filename} -T {htk_trace} "
                       "{global_script} {output_dictionary_filename} "
                       "{dictionary_filenames}".format(htk_trace=htk_trace,
                                                       global_script=global_script,
                                                       dictionary_filenames=' '.join([filename
                                                                                      for filename
                                                                                      in
                                                                                      input_dictionary_in_nfc_filenames
                                                                                      if
                                                                                      os.path.getsize(filename) > 0]),
                                                       output_dictionary_filename=output_dictionary_filename,
                                                       log_filename=log_filename))

    os.remove(log_filename)
    for filename in input_dictionary_in_nfc_filenames:
        os.remove(filename)

    htk_to_utf8normalize(output_dictionary_filename)

    return


def prepare_dictionary(words_filename, dictionary_filename, output_dictionary_filename, htk_trace,
                       global_script_filename=None, add_sent_silence=True, additional_dictionary_filenames=[]):
    startendsilence_dictionary_filename = config.project_path('etc', 'startendsilence.dict')
    global_script = ""
    if global_script_filename:
        global_script = u"-g {}".format(global_script_filename)

    temp_dict_filename = utf8_normalize(dictionary_filename, to_upper=False)
    temp_words_filename = utf8_normalize(words_filename, to_upper=True)

    if add_sent_silence:
        with codecs.open(temp_words_filename, 'a', 'utf-8') as f:
            startendsilence = []
            with codecs.open(startendsilence_dictionary_filename, 'r', 'utf-8') as startendsilence_file:
                for line in startendsilence_file:
                    if line.strip():
                        startendsilence.append(line.strip().split(None, 1)[0])

            f.writelines([u'{}\n'.format(token) for token in startendsilence])

    log_filename = config.temp_path('hdman.log')
    config.htk_command("HDMan -A -l {log_filename} -T {htk_trace} "
                       "-w {words_filename} {global_script} "
                       "{output_dictionary_filename} {dictionary_filename} "
                       "{startendsilence_dictionary_filename} "
                       "{additional_dictionary_filenames}".format(htk_trace=htk_trace,
                                                                  words_filename=temp_words_filename,
                                                                  global_script=global_script,
                                                                  dictionary_filename=temp_dict_filename,
                                                                  startendsilence_dictionary_filename=startendsilence_dictionary_filename,
                                                                  output_dictionary_filename=output_dictionary_filename,
                                                                  log_filename=log_filename,
                                                                  additional_dictionary_filenames=' '.join(
                                                                      additional_dictionary_filenames)))

    # Parse the log file to look for missing words
    with codecs.open(log_filename, 'r', 'utf-8') as f:
        lines = f.read()

    missing_words = re.search(r'Missing Words\n-------------\n((.+\s)+)', lines, re.MULTILINE)
    if missing_words:
        words = missing_words.groups()[0].strip().split()
        raise ValueError(u"The following words could not be found by "
                         u"HTK in the pronunciation dictionary: \n{}".format(u'\n'.join(words)))

    os.remove(log_filename)
    os.remove(temp_words_filename)
    os.remove(temp_dict_filename)

    return


def parse_instance_count(instance_count, speaker_total_count):
    """This parses the instance count dictionary
    (that may contain floats from 0.0 to 1.0 representing a percentage)
    and converts it to actual instance count.
    """

    # Use all the instances of a speaker unless specified
    result = copy.copy(speaker_total_count)
    for speaker_id, count in instance_count.items():
        speaker_id = str(speaker_id)
        speaker_total = speaker_total_count.get(speaker_id, 0)

        if type(count) == float and 0.0 <= count <= 1.0:
            result[speaker_id] = int(speaker_total * count)

        else:
            result[speaker_id] = int(count)

    return result


def glob_recordings(recording_dirs, recording_patterns):
    for recording_dir in recording_dirs:
        if not os.path.isdir(recording_dir):
            logging.error("Recording directory {} supplied by user does not exist.".format(recording_dir))
            sys.exit(1)

    # Normalize to NFC (because Apple decided to be "original" and use NFD for their filesystem)
    recordings = []
    for recording_pattern in recording_patterns:
        for recording_dir in recording_dirs:
            recordings += [unicodedata.normalize(UTF8_NORMALIZATION, f) for f in
                           recursive_glob(unicode(recording_dir), recording_pattern)]

    return list(set(recordings))


def select_recording_set(recording_dirs, speaker_recording_instance_count, recordingset_filename, possible_words=None,
                         unaccepted_recordings=[]):
    unaccepted_recordings = set(unaccepted_recordings)

    recordings = recordings_possible = glob_recordings(recording_dirs, ['*.wav'])

    # Check if the words present in the recordings are all in the list of possible words if supplied
    if possible_words is not None:
        possible_words = set(possible_words)

        def all_words_possible(recording):
            return set([w.upper() for w in get_words_from_recording(recording)]) <= possible_words

        recordings_possible = filter(all_words_possible, recordings_possible)
        recordings_rejected = set(recordings) - set(recordings_possible)
        if len(recordings_rejected) > 0:
            logging.warning(u"Some recordings had to be rejected because "
                            "some of their words could not be found "
                            "in the possible words list: \n{}".format(u'\n'.join(recordings_rejected)))

    # Convert the speaker IDs to strings (to match the type of the IDs parsed from the recording filename)
    speaker_total_count = dict([(k, len(list(v)))
                                for k, v
                                in itertools.groupby(sorted(recordings_possible,
                                                            key=get_speaker_from_recording),
                                                     key=get_speaker_from_recording)])

    logging.debug('Speaker total count: {}'.format(speaker_total_count))
    speaker_recording_instance_count = parse_instance_count(speaker_recording_instance_count, speaker_total_count)

    logging.debug('Speaker recording instance count: {}'.format(speaker_recording_instance_count))

    recordings = filter(lambda x: x not in unaccepted_recordings, recordings_possible)

    # Check that there are no two recordings with the same basename (since this is used as ID in MLFs)
    recordings_ids_duplicates = {}
    for k, v in itertools.groupby(sorted(recordings, key=os.path.basename), key=os.path.basename):
        values = list(v)
        if len(values) > 1:
            recordings_ids_duplicates[k] = values

    if len(recordings_ids_duplicates) > 0:
        raise ValueError(
            "The recording filename basename must be unique since "
            "this is used as ID in the MLFs of HTK. Duplicated: {}".format(
                recordings_ids_duplicates))

    groups = itertools.groupby(sorted(recordings,
                                      key=get_speaker_from_recording),
                               key=get_speaker_from_recording)

    speaker_groups = [(k, list(v)) for k, v in groups]

    list_recording = []

    num_recording_instances = collections.defaultdict(lambda: sys.maxint)
    if speaker_recording_instance_count is not None:
        num_recording_instances = collections.defaultdict(lambda: 0)
        num_recording_instances.update(speaker_recording_instance_count)

    for i, (speaker_id, speaker_recordings) in enumerate(speaker_groups):
        num_instances = num_recording_instances[speaker_id]
        num_available = len(speaker_recordings)

        if num_instances > num_available:
            logging.warning(
                "Requested {} instances of speaker {} but only {} recordings are available.".format(num_instances,
                                                                                                    speaker_id,
                                                                                                    num_available))

        recording_instances = random.sample(speaker_recordings, min(num_instances, num_available))
        list_recording += recording_instances

    if len(list_recording) == 0:
        raise ValueError("No recordings could be selected.  Consider allowing the selection "
                         "of more recordings by removing filters --no-filter-training-recordings "
                         "or --no-filter-adaptation-recordings.")

    with codecs.open(recordingset_filename, 'w', 'utf-8') as f:
        f.write(u'\n'.join(list_recording))

    return list_recording


def create_feature_files(list_recordings, feature_directory, inputoutput_feature_file, output_feature_file,
                         feature_extractor, feature_extractor_parameters):
    list_inputoutput_feature = [(u'{}'.format(wavFile),
                                 u'{}.mfc'.format(os.path.join(feature_directory,
                                                               os.path.splitext(os.path.basename(wavFile))[0])))
                                for wavFile in list_recordings]

    # Create the feature space-separated input/output file
    with codecs.open(inputoutput_feature_file, 'w', 'utf-8') as f:
        f.writelines([u'"{}" "{}"\n'.format(wavFile, featureFile) for wavFile, featureFile in list_inputoutput_feature])

    # Create the file that contains the list of output feature files
    with codecs.open(output_feature_file, 'w', 'utf-8') as f:
        f.writelines([u'"{}"\n'.format(featureFile, ) for wavFile, featureFile in list_inputoutput_feature])

    cmd = u"{} {} {}".format(feature_extractor, feature_extractor_parameters, inputoutput_feature_file)
    logging.info(cmd)
    config.sh_command(cmd)

    return


def force_align(model_directory,
                inputoutput_feature_filename,
                symbollist_filename,
                word_mlf_filename,
                dictionary_filename,
                aligned_filename,
                output_words=False,
                output_likelihoods=False,
                output_times=False,
                level='word',
                boundary_silence=True,
                htk_trace=0):
    _aligned_file = aligned_filename
    if aligned_filename == word_mlf_filename:
        _aligned_file = config.temp_path('word_mlf_aligned.mlf')

    temp_dictionary_filename = utf8_normalize(dictionary_filename)

    output_arg = '{}{}{}'.format('' if output_likelihoods else 'S',
                                 '' if output_words else 'W',
                                 '' if output_times else 'T')

    levels = {'word': '',
              'state': '-f'}

    if level not in levels:
        raise ValueError('Only the following alignment levels are supported {}'.format(levels.keys()))

    level_arg = '{}'.format(levels[level])

    boundary_silence_str = '-b silence' if boundary_silence else ''

    config.htk_command("HVite -o '{}' {} -a -A -D -T {}"
                       " {} -S {}"
                       " -H {} -H {} -I {} -y lab"
                       " -i {} {} {}".format(output_arg,
                                             level_arg,
                                             htk_trace,
                                             boundary_silence_str,
                                             inputoutput_feature_filename,
                                             os.path.join(model_directory, 'macros'),
                                             os.path.join(model_directory, 'hmmdefs'),
                                             word_mlf_filename,
                                             _aligned_file,
                                             temp_dictionary_filename,
                                             symbollist_filename))
    os.remove(temp_dictionary_filename)

    if aligned_filename == word_mlf_filename:
        shutil.copyfile(_aligned_file, aligned_filename)
        os.remove(_aligned_file)

    return


def create_prototype_model(output_feature_filename,
                           output_prototype_filename,
                           state_stay_probabilities=[0.5]):
    with codecs.open(output_feature_filename, 'r', 'utf-8') as output_feature_file:
        example_feature_filename = shlex_split(output_feature_file.readline())[0]

    meta = htk.load_parameter(example_feature_filename, only_header=True)
    sample_dimension = int(meta.samp_size / 4)
    parameter_kind_base = meta.parm_kind_base
    parameter_kind_options = meta.parm_kind_opts

    proto_model = htk_model_utils.create_prototype(sample_dimension,
                                                   parameter_kind_base=parameter_kind_base,
                                                   parameter_kind_options=parameter_kind_options,
                                                   state_stay_probabilities=state_stay_probabilities)

    htk.save_model(proto_model, output_prototype_filename)


def create_flat_start_model(feature_filename,
                            state_stay_probabilities,
                            symbol_list,
                            output_model_directory,
                            output_prototype_filename,
                            htk_trace):
    """
    Creates a flat start model by using HCompV to compute the global mean and variance.
    Then uses these global mean and variance to create an N-state model for each symbol in the given list.

    :param feature_filename: The filename containing the audio and feature file pairs
    :param output_model_directory: The directory where to write the created model
    :param output_prototype_filename: The prototype model filename
    :param htk_trace: Trace level for HTK
    :rtype : None
    """
    # Create a prototype model
    create_prototype_model(feature_filename,
                           output_prototype_filename,
                           state_stay_probabilities=state_stay_probabilities)

    # Compute the global mean and variance
    config.htk_command("HCompV -A -D -T {} -f 0.01 "
                       "-S {} -m -o {} -M {} {}".format(htk_trace,
                                                        feature_filename,
                                                        'proto',
                                                        output_model_directory,
                                                        output_prototype_filename))

    # Create an hmmdefs using the global mean and variance for all states and symbols
    # Duplicate the model 'proto' -> symbol_list
    proto_model_filename = config.path(output_model_directory, 'proto')
    model = htk.load_model(proto_model_filename)
    model = htk_model_utils.map_hmms(model, {'proto': symbol_list})

    # vFloors -> macros
    vfloors_filename = config.path(output_model_directory, 'vFloors')
    variance_model = htk.load_model(vfloors_filename)

    model['macros'] += variance_model['macros']

    macros, hmmdefs = htk_model_utils.split_model(model)

    htk.save_model(macros, config.path(output_model_directory, 'macros'))
    htk.save_model(hmmdefs, config.path(output_model_directory, 'hmmdefs'))


def htk_mlf_symbol_replace(mlf,
                           replace_method=lambda x: x,
                           compress=True):
    new_mlf = {}
    for recording, label_seqs in mlf.items():
        new_label_seqs = []
        for label_seq in label_seqs:
            new_label_seq = []
            for label in label_seq:
                new_symbol = replace_method(label.symbol)
                label = label._replace(symbol=new_symbol)
                if compress \
                        and len(new_label_seq) \
                        and label.symbol == new_label_seq[-1].symbol:
                    # If we want to compress
                    # the new label sequence is not empty
                    # the new symbol is equal to the previous one
                    # we update the previous end time
                    new_label_seq[-1] = new_label_seq[-1]._replace(end=label.end)
                else:
                    # otherwise we append the new label
                    new_label_seq.append(label)

            new_label_seqs.append(new_label_seq)

        new_mlf[recording] = new_label_seqs

    return new_mlf


def train_speech_prior(model_directory,
                       component_count,
                       output_directory,
                       pronunciation_dictionary_filename,
                       list_recordings,
                       feature_filename,
                       speech_prior_mlf_filename,
                       htk_trace):
    models_directory = config.path('tmp/speech_prior', create=True)

    # Force align using the MLF triphone+sp files
    symbollist_filename = config.path(model_directory, 'symbollist')

    aligned_mlf_filename = config.path(models_directory, 'speech_prior_phone_aligned.mlf')
    reference_filename = config.path(models_directory, 'speech_prior_word.mlf')
    create_recordings_mlf(list_recordings,
                          pronunciation_dictionary_filename=pronunciation_dictionary_filename,
                          mlf_word_file=reference_filename,
                          htk_trace=htk_trace)

    logging.debug('Create a state alignment with the model for the silence/speech labels')
    force_align(model_directory,
                feature_filename,
                symbollist_filename,
                reference_filename,
                pronunciation_dictionary_filename,
                aligned_mlf_filename,
                output_times=True,
                output_likelihoods=False,
                output_words=False,
                level='state',
                htk_trace=htk_trace)

    # Modify the MLF files to only contain speech and sp symbols
    recognised_mlf = htk.load_mlf(aligned_mlf_filename, utf8_normalization=UTF8_NORMALIZATION)

    def only_silence_speech(x):
        if x.startswith('sil[') or x.startswith('sp['):
            return 'sil'
        else:
            return 'speech'

    replaced_mlf = htk_mlf_symbol_replace(recognised_mlf,
                                          replace_method=only_silence_speech,
                                          compress=True)

    htk.save_mlf(replaced_mlf, speech_prior_mlf_filename)

    # Initialize a single-state HMM model for speech
    logging.debug('Create flat start prior model')
    prior_prototype_filename = config.path(models_directory, 'proto.1state.mfc')
    model_directory = config.path(models_directory, 'prior.0', create=True)
    state_stay_probabilities = [0.5]
    symbol_list = ['speech', 'sil']
    create_flat_start_model(feature_filename,
                            state_stay_probabilities,
                            symbol_list,
                            model_directory,
                            prior_prototype_filename,
                            htk_trace)

    symbols_filename = config.path(models_directory, 'prior_symbols')
    with codecs.open(symbols_filename, 'w', 'utf-8') as f:
        f.write('\n'.join(symbol_list) + '\n')

    # Add GMM components and iterate
    logging.debug('Train prior model')
    # Reestimate parameters for monophone models (5 iterations)
    iterations = [1, 2, 3, 4, 5]
    for iteration in iterations:
        previous_model_directory = model_directory
        model_directory = config.path(models_directory, 'prior.{:03d}mix.{}'.format(1, iteration), create=True)

        config.htk_command("HERest -A -D -T {} -S {} -d {} "
                           "-H {} -H {} -I {} -M {} {}".format(htk_trace,
                                                               feature_filename,
                                                               previous_model_directory,
                                                               os.path.join(
                                                                   previous_model_directory,
                                                                   'macros'),
                                                               os.path.join(
                                                                   previous_model_directory,
                                                                   'hmmdefs'),
                                                               speech_prior_mlf_filename,
                                                               model_directory,
                                                               symbols_filename))

        # Copy the necessary to the model directory
        shutil.copy(symbols_filename, os.path.join(model_directory, 'symbollist'))

    # Increase the number of state mixtures
    mixtures = np.asarray(np.round(np.exp(np.linspace(np.log(2), np.log(component_count), 5))), dtype=np.int)
    for mixture in mixtures:
        previous_model_directory = model_directory
        model_directory = config.path(models_directory, 'prior.{:03d}mix.0'.format(mixture), create=True)

        script_filename = config.path(models_directory, 'mix{}.hed'.format(mixture))
        with codecs.open(script_filename, 'w', 'utf-8') as f:
            f.write('MU {} {{*.state[2-20].mix}}\n'.format(mixture))

        config.htk_command('HHEd -A -D -T {} -d {} '
                           '-H {} -H {} -M {} {} {}'.format(htk_trace,
                                                            previous_model_directory,
                                                            os.path.join(
                                                                previous_model_directory,
                                                                'macros'),
                                                            os.path.join(
                                                                previous_model_directory,
                                                                'hmmdefs'),
                                                            model_directory,
                                                            script_filename,
                                                            symbols_filename))

        # Copy the necessary to the model directory
        shutil.copy(symbols_filename, os.path.join(model_directory, 'symbollist'))

        iterations = [1, 2, 3, 4]
        for iteration in iterations:
            previous_model_directory = model_directory
            model_directory = config.path(models_directory, 'prior.{:03d}mix.{}'.format(mixture, iteration), create=True)

            config.htk_command("HERest -A -D -T {} -S {} -d {} "
                               "-H {} -H {} -I {} -M {} {}".format(htk_trace,
                                                                   feature_filename,
                                                                   previous_model_directory,
                                                                   os.path.join(
                                                                       previous_model_directory,
                                                                       'macros'),
                                                                   os.path.join(
                                                                       previous_model_directory,
                                                                       'hmmdefs'),
                                                                   speech_prior_mlf_filename,
                                                                   model_directory,
                                                                   symbols_filename))

            # Copy the necessary to the model directory
            shutil.copy(symbols_filename, os.path.join(model_directory, 'symbollist'))

    # Copy the final model files to the output directory
    shutil.copy(os.path.join(model_directory, 'macros'), os.path.join(output_directory, 'macros'))
    shutil.copy(os.path.join(model_directory, 'hmmdefs'), os.path.join(output_directory, 'hmmdefs'))
    shutil.copy(os.path.join(model_directory, 'symbollist'), os.path.join(output_directory, 'symbollist'))


def create_recordings_mlf(list_recordings,
                          pronunciation_dictionary_filename=None,
                          force_align_model_directory=None,
                          force_align_output_feature_filename=None,
                          force_align_symbollist_filename=None,
                          phones_filename=None,
                          phonessp_filename=None,
                          mlf_word_file=None,
                          mlf_phone_file=None,
                          mlf_phonesp_file=None,
                          mlf_tri_file=None,
                          mlf_trisp_file=None,
                          phone_script_filename=None,
                          phonesp_script_filename=None,
                          triphones_filename=None,
                          triphonessp_filename=None,
                          htk_trace=0):
    temp_files = []

    def create_and_track(x):
        f = config.temp_path(x)
        temp_files.append(f)
        return f

    _mlf_word_file = mlf_word_file or create_and_track('word.mlf')

    _mlf_phone_file = mlf_phone_file or create_and_track('phone.mlf')
    _mlf_phonesp_file = mlf_phonesp_file or create_and_track('phonesp.mlf')
    _phones_filename = phones_filename or create_and_track('phones')
    _phonessp_filename = phonessp_filename or create_and_track('phonessp')

    _mlf_tri_file = mlf_tri_file or create_and_track('tri.mlf')
    _mlf_trisp_file = mlf_trisp_file or create_and_track('trisp.mlf')
    _triphones_filename = triphones_filename or create_and_track('triphones')
    _triphonessp_filename = triphonessp_filename or create_and_track('triphonessp')

    phone_script_filename = phone_script_filename or config.project_path('etc', 'mkphones1.led')
    phonesp_script_filename = phonesp_script_filename or config.project_path('etc', 'mkphones1sp.led')

    compute_triphone = any([triphonessp_filename, triphones_filename, mlf_tri_file, mlf_trisp_file])
    compute_phone = any([compute_triphone, phonessp_filename, phones_filename, mlf_phone_file, mlf_phonesp_file])
    compute_word = any([compute_phone, mlf_word_file])

    if compute_word:
        logging.info("building word level MLFs")
        with codecs.open(_mlf_word_file, 'w', 'utf-8') as f:
            f.write("#!MLF!#\n")
            for recording in list_recordings:
                words = get_words_from_recording(recording)

                f.write(u'"*/{}.lab"\n'.format(os.path.splitext(os.path.basename(recording))[0]))
                f.writelines([u'{}\n'.format(word.upper()) for word in words])
                f.write(u'.\n\n')

                # Perform a force alignment using the passed model filename
    if force_align_model_directory \
            or force_align_output_feature_filename \
            or force_align_symbollist_filename:
        if not force_align_model_directory \
                or not force_align_output_feature_filename \
                or not force_align_symbollist_filename:
            raise ValueError("The model directory, the feature file and the symbol list"
                             " (wordlist, phonelist or tiedlist) are required when aligning.")

        force_align(force_align_model_directory,
                    force_align_output_feature_filename,
                    force_align_symbollist_filename,
                    _mlf_word_file,
                    pronunciation_dictionary_filename,
                    _mlf_word_file,
                    htk_trace)

    if compute_phone:
        if pronunciation_dictionary_filename is None:
            raise ValueError("A pronunciation dictionary file is required to generate phone level MLF.")

        temp_pronunciation_dictionary_filename = utf8_normalize(pronunciation_dictionary_filename)
        temp_files.append(temp_pronunciation_dictionary_filename)

        logging.info("building phone level MLFs")
        config.htk_command(
            "HLEd -A -C {} -D -T {} -l '*' -d {} -n {} -i {} {} {}".format(config.project_path('etc', 'config_empty'),
                                                                           htk_trace,
                                                                           temp_pronunciation_dictionary_filename,
                                                                           _phones_filename,
                                                                           _mlf_phone_file,
                                                                           phone_script_filename,
                                                                           _mlf_word_file))

        logging.info("building phone level and pause MLFs")
        config.htk_command(
            "HLEd -A -C {} -D -T {} -l '*' -d {} -n {} -i {} {} {}".format(config.project_path('etc', 'config_empty'),
                                                                           htk_trace,
                                                                           temp_pronunciation_dictionary_filename,
                                                                           _phonessp_filename,
                                                                           _mlf_phonesp_file,
                                                                           phonesp_script_filename,
                                                                           _mlf_word_file))

    if compute_triphone:
        logging.info("building triphone level MLFs")
        config.htk_command(
            "HLEd -A -C {} -D -T {} -l '*' -n {} -i {} {} {}".format(config.project_path('etc', 'config_empty'),
                                                                     htk_trace,
                                                                     _triphones_filename,
                                                                     _mlf_tri_file,
                                                                     config.project_path('etc/mktri.led'),
                                                                     _mlf_phone_file))

        logging.info("building triphone level and pause MLFs")
        config.htk_command(
            "HLEd -A -C {} -D -T {} -l '*' -n {} -i {} {} {}".format(config.project_path('etc', 'config_empty'),
                                                                     htk_trace,
                                                                     _triphonessp_filename,
                                                                     _mlf_trisp_file,
                                                                     config.project_path('etc/mktri.led'),
                                                                     _mlf_phonesp_file))

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return


"""
def create_triphone_dictionary(dictionary_filename,
                               triphone_dictionary_filename,
                               output_triphone_sybmol_filename = None,

                               htk_trace = 0):
    config.htk_command("HDMan -A -D -T {} -b sp "
                       "-n {} -g {} -l {} {} {}".format(htk_trace,
                                                        triphonehtk_filename,
                                                        config.project_path('etc',
                                                                            'global.tri.ded'),
                                                        log_filename,
                                                        pronunciation_triphone_dictionary_filename,
                                                        pronunciation_dictionary_filename))
"""


def get_dictionary_homophones(dictionary, words=None):
    inverse_dictionary = collections.defaultdict(set)
    for word, pronunciations in dictionary.items():
        if words is not None and word not in words:
            continue

        for pronunciation in pronunciations:
            inverse_dictionary[pronunciation].add(word)

    # TODO: Take possible words from the reference and recognition files
    # possible_words = {}
    # homophones = [(list(words)) for pronunciation, words
    # in inverse_dictionary.items() if len(words & possible_words) > 1]
    homophones = [list(words) for pronunciation, words in inverse_dictionary.items() if len(words) > 1]
    return homophones


def evaluate_model(reference_filename,
                   symbollist_filename,
                   recognition_filename,
                   dictionary_filename='',
                   htk_trace=0):
    ignore_arguments = '-e "???" "SENT-START" -e "???" "SENT-END" -e "???" sil -e "???" sp'

    # Only take words in the reference and recognition for the homophones
    mlf_ref = htk.load_mlf(reference_filename, UTF8_NORMALIZATION)
    mlf_rec = htk.load_mlf(recognition_filename, UTF8_NORMALIZATION)

    all_words = set()

    for hypotheses in mlf_ref.values() + mlf_rec.values():
        for label_seq in hypotheses:
            for label in label_seq:
                all_words.add(label.symbol)

    # Set equivalents based on homonyms
    homophones_arguments = u''
    if os.path.isfile(dictionary_filename):
        dictionary = parse_dictionary(dictionary_filename)
        homophones = get_dictionary_homophones(dictionary, words=all_words)
        for word_list in homophones:
            for word in word_list[1:]:
                homophones_arguments += u' -e "{}" "{}"'.format(word_list[0], word)

    cmd = u'HResults -T {htk_trace} -A {ignore_arguments} ' \
          u'{homophones_arguments} -I {reference_filename} ' \
          u'{symbollist_filename} {recognition_filename}'.format(reference_filename=reference_filename,
                                                                 symbollist_filename=symbollist_filename,
                                                                 recognition_filename=recognition_filename,
                                                                 ignore_arguments=ignore_arguments,
                                                                 homophones_arguments=homophones_arguments,
                                                                 htk_trace=max(1, htk_trace))
    config.htk_command(cmd)


def recognise_model(feature_filename,
                    symbollist_filename,
                    model_directory,
                    recognition_filename,
                    pronunciation_dictionary_filename,
                    list_words_filename='',
                    cmllr_directory=None,
                    tokens_count=None,
                    hypotheses_count=1,
                    htk_trace=0):
    """
        Perform recognition using a model and assuming a single word language.

        If the list_words_filename is == '' then all the words in the dictionary are used as language words.
    """
    # Normalize UTF-8 to avoid Mac problems
    temp_dictionary_filename = utf8_normalize(pronunciation_dictionary_filename)

    # Create language word list
    if list_words_filename:
        list_words = parse_wordlist(list_words_filename)
    else:
        list_words = sorted(parse_dictionary(temp_dictionary_filename).keys())

    # Create language model
    temp_directory = config.project_path('tmp', create=True)
    grammar_filename = config.path(temp_directory, 'grammar_words')
    wdnet_filename = config.path(temp_directory, 'wdnet')
    logging.debug('Create language model')
    create_language_model(list_words,
                          grammar_filename,
                          wdnet_filename)

    # Handle the Adaptation parameters
    cmllr_arguments = ''
    if cmllr_directory:
        if not os.path.isdir(cmllr_directory):
            logging.error('CMLLR adapatation directory not found: {}'.format(cmllr_directory))

        cmllr_arguments = "-J {} mllr2 -h '*/*_s%.mfc' -k -J {}".format(
            os.path.abspath(config.path(cmllr_directory, 'xforms')),
            os.path.abspath(config.path(cmllr_directory, 'classes')))
    # Handle the N-Best parameters
    hypotheses_count = hypotheses_count or 1
    tokens_count = tokens_count or int(math.ceil(hypotheses_count / 5.0))
    if hypotheses_count == 1 and tokens_count == 1:
        nbest_arguments = ""
    else:
        nbest_arguments = "-n {tokens_count} {hypotheses_count} ".format(tokens_count=tokens_count,
                                                                         hypotheses_count=hypotheses_count)

    # Run the HTK command
    config.htk_command("HVite -A -l '*' -T {htk_trace} "
                       "-H {model_directory}/macros -H {model_directory}/hmmdefs "
                       "-i {recognition_filename} -S {feature_filename} "
                       "{cmllr_arguments} -w {wdnet_filename} "
                       "{nbest_arguments} "
                       "-p 0.0 -s 5.0 "
                       "{pronunciation_dictionary_filename} "
                       "{symbollist_filename}".format(htk_trace=htk_trace,
                                                      model_directory=model_directory,
                                                      recognition_filename=recognition_filename,
                                                      feature_filename=feature_filename,
                                                      symbollist_filename=symbollist_filename,
                                                      nbest_arguments=nbest_arguments,
                                                      pronunciation_dictionary_filename=temp_dictionary_filename,
                                                      wdnet_filename=wdnet_filename,
                                                      cmllr_arguments=cmllr_arguments))

    # Remove temporary files
    os.remove(temp_dictionary_filename)
    os.remove(wdnet_filename)
    os.remove(grammar_filename)


def create_language_model(list_words,
                          grammar_filename,
                          wdnet_filename):
    """Create a language model (wdnet) for HTK using a very simple single-word grammar"""

    # Create the temporary grammar file
    grammar = u''
    grammar += u'$possible_words = '
    grammar += u' | '.join([word for word in list_words])
    grammar += u';\n'

    grammar += u'( [SENT-START] ( $possible_words ) [SENT-END] )\n'

    with codecs.open(grammar_filename, 'w', 'utf-8') as f:
        f.write(grammar)

    # Use HParse to create the wdnet file
    config.htk_command('HParse -A {} {}'.format(grammar_filename, wdnet_filename))

    return
