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
import copy

__author__ = 'rmarxer'


import numpy as np
import collections

from . import htk_model


def create_default():
    return collections.defaultdict(lambda: None)


def create_vector(vector):
    return {'dim': vector.size, 'vector': vector}


def create_square_matrix(mat):
    return {'dim': mat.shape[0], 'matrix': mat}


def create_transition(state_stay_probabilites=[0.6, 0.6, 0.7]):
    state_probs = np.array(state_stay_probabilites)
    n_states = state_probs.size + 2
    transitions = np.zeros((n_states, n_states))
    transitions[0, 1] = 1.0
    for i, p in enumerate(state_probs):
        transitions[i + 1, i + 1] = p
        transitions[i + 1, i + 2] = 1 - p

    return create_square_matrix(transitions)


def create_parameter_kind(base=None, options=[]):
    result = create_default()
    result['base'] = base
    result['options'] = options
    return result


def create_options(vector_size=None,
                   parameter_kind=None):

    macro = create_default()

    options = []
    if vector_size:
        option = create_default()
        option['vector_size'] = vector_size
        options.append(option)

    if parameter_kind:
        option = create_default()
        option['parameter_kind'] = parameter_kind
        options.append(option)

    macro['options'] = {'definition': options}

    return macro


def create_gmm(means, variances, gconsts=None, weights=None):
    mixtures = []

    if means.ndim == 1:
        means = means[None, :]
        variances = variances[None, :]

    gmm = create_default()

    for i in range(means.shape[0]):
        mixture = create_default()
        mixture['pdf'] = create_default()
        mixture['pdf']['mean'] = create_vector(means[i])
        mixture['pdf']['covariance'] = create_default()
        mixture['pdf']['covariance']['variance'] = create_vector(variances[i])

        if gconsts is not None:
            mixture['pdf']['gconst'] = gconsts[i]

        if weights is not None:
            mixture['weight'] = weights[i]

        mixtures.append(mixture)

    stream = create_default()
    stream['mixtures'] = mixtures

    gmm['streams'] = [stream]

    return gmm


def create_hmm(states, transition, name=None):
    hmm = create_default()
    hmm['name'] = name
    hmm['definition'] = create_default()

    hmm['definition']['state_count'] = len(states) + 2
    hmm['definition']['states'] = []
    for i, state in enumerate(states):
        hmm_state = create_default()

        hmm_state['index'] = i + 2
        hmm_state['state'] = state

        hmm['definition']['states'].append(hmm_state)

    hmm['definition']['transition'] = transition

    return hmm


def create_model(macros, hmms):
    model = create_default()

    model['macros'] = macros
    model['hmms'] = hmms

    return model


def create_prototype(sample_dimension,
                     parameter_kind_base='user',
                     parameter_kind_options=[],
                     state_stay_probabilities=[0.6, 0.6, 0.7]):
    """Create a prototype HTK model file using a feature file.
    """
    parameter_kind = create_parameter_kind(base=parameter_kind_base,
                                           options=parameter_kind_options)

    transition = create_transition(state_stay_probabilities)

    state_count = len(state_stay_probabilities)

    states = []
    for i in range(state_count):
        state = create_gmm(np.zeros(sample_dimension),
                           np.ones(sample_dimension),
                           weights=None,
                           gconsts=None)

        states.append(state)

    hmms = [create_hmm(states, transition)]
    macros = [create_options(vector_size=sample_dimension,
                             parameter_kind=parameter_kind)]

    model = create_model(macros, hmms)

    return model


def split_model(model):
    macros_model = copy.deepcopy(model)
    hmmdefs_model = copy.deepcopy(model)

    filter_macros = lambda x: x['transition'] is None and x['state'] is None

    macros_model['macros'] = filter(filter_macros, model['macros'])
    macros_model['hmms'] = []

    hmmdefs_model['macros'] = filter(lambda x: not filter_macros(x), model['macros'])

    return macros_model, hmmdefs_model


def map_hmms(input_model, mapping):
    """Create a new HTK HMM model given a model and a mapping dictionary.

    :param input_model: The model to transform of type dict
    :param mapping: A dictionary from string -> list(string)
    :return: The transformed model of type dict
    """

    output_model = copy.copy(input_model)

    o_hmms = []
    for i_hmm in input_model['hmms']:
        i_hmm_name = i_hmm['name']
        o_hmm_names = mapping.get(i_hmm_name, [i_hmm_name])

        for o_hmm_name in o_hmm_names:
            o_hmm = copy.copy(i_hmm)
            o_hmm['name'] = o_hmm_name

            o_hmms.append(o_hmm)

    output_model['hmms'] = o_hmms

    return output_model


def transform_to_semi_hmm(model, repetition_count=4, min_repetitions=1, max_repetitions=np.inf):
    """This function expands and HMM model into a semi HMM model by multiplying the number of states repetition_coutn:
    times and modify the transition matrices accordingly to only allow between min_repetitions: and max_repetitions:
    repetitions.

    :param model: The model to transform in place
    :param repetition_count: The amount of repeated occupations of a state to model
    :param min_repetitions: The minimum amount of repeated occupations of a state to allow
    :param max_repetitions: The maximum amount of repeated occupations of a state to allow
    """

    def transform_transition(definition):
        """Transform in place the definition of the transition element.
        :param definition: The transition element
        """
        dim = definition['dim']
        matrix = definition['matrix']

        new_dim = (definition['dim'] - 2) * repetition_count + 2

        new_matrix = np.zeros((new_dim, new_dim))

        new_matrix[0, 1] = matrix[0, 1]

        def state_repeat_to_ind(sta, rep):
            """Return the index corresponding to the state sta: and repetition rep:.
            States and repetitions start at 0 (state 0 corresponds to the initial dummy state).

            :param sta: The state for which to get the index.
            :param rep: The repetition for which to get the index
            :return: The index in the transition matrix
            """
            if sta == 0:
                return 0

            return ((sta - 1) * repetition_count + rep) + 1

        def probability(source_state, source_repeat, target_state, target_repeat):
            """Return the computed probability of the transition,
            given the source (state source_state:, repeat source_repeat:)
            and the target (state target_state:, repeat target_repeat:).
            Currently this simply keeps the probabilistic model unchanged.

            :param source_state: The source state
            :param source_repeat: The source repetition
            :param target_state: The target state
            :param target_repeat: The target repetition
            """
            return matrix[source_state, target_state]

        for state in range(1, dim-1):
            for repeat in range(min_repetitions-1, min(max_repetitions, repetition_count)):
                source = state_repeat_to_ind(state, repeat)

                # Case in which we advance to the next state
                target = state_repeat_to_ind(state+1, 0)
                prob = probability(state, repeat, state+1, 0)
                new_matrix[source, target] = prob

                if repeat == repetition_count - 1:
                    if np.isinf(max_repetitions):
                        # Case in which we repeat the last state that we model (self-transition)
                        target = source
                        prob = probability(state, repeat, state, repeat)
                        new_matrix[source, target] = prob

                else:
                    # Case in which repeat the state again
                    target = state_repeat_to_ind(state, repeat+1)
                    prob = probability(state, repeat, state, repeat+1)
                    new_matrix[source, target] = prob

        new_matrix[:-1, :] /= new_matrix.sum(axis=1)[:-1, np.newaxis]

        definition['dim'] = new_dim
        definition['matrix'] = new_matrix
        return definition

    # Go through all hmm models
    transition_references = set()
    for hmm in model['hmms']:
        new_statecount = (hmm['definition']['state_count'] - 2) * repetition_count + 2
        hmm['definition']['state_count'] = new_statecount

        # Expand states mantaining the expanded states tied
        new_states = []
        index = None
        for state in sorted(hmm['definition']['states'], key=lambda x: x['index']):
            index = index or state['index']
            for repetition in range(repetition_count):
                new_state = copy.copy(state)
                new_state['index'] = index
                new_states.append(new_state)
                index += 1

        hmm['definition']['states'] = new_states

        # Transform the transition matrices
        if isinstance(hmm['definition']['transition'], collections.Mapping):
            # Definition: transform now
            transform_transition(hmm['definition']['transition']['definition'])
        else:
            # Reference: collect it and transform them all later
            transition_references.add(hmm['definition']['transition'])

    for reference in transition_references:
        definitions = filter(lambda x: x['transition'] and x['transition']['name'] == reference, model['macros'])

        if not len(definitions) == 1:
            logging.warning('Invalid number of definitions ({}) for transition macro {}.'
                            '  Expecting 1 definition.'.format(len(definitions), reference))

        for definition in definitions:
            transform_transition(definition['transition']['definition'])


def main():
    model = htk_model.load_model('test')
    macros, hmmdefs = split_model(model)
    print(htk_model.model_to_json(macros, indent=4))
    print(htk_model.model_to_json(hmmdefs, indent=4))

if __name__ == '__main__':
    main()