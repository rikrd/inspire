#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections

import copy
import sys
import json

import numpy


def _combine_dicts(*args):
    all_dict = []
    for arg in args:
        all_dict += list(arg.items())

    return dict(all_dict)


class Operation:
    def __init__(self, op_code, index, from_symbol, to_symbol):
        self.op_code = op_code
        self.index = index
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol

    def to_primitive(self):
        return self.__dict__

    def __repr__(self):
        return u'({}, {}, {}, {})'.format(self.op_code, self.index, self.from_symbol, self.to_symbol)


class Script:
    def __init__(self, i, j, path=None):
        self.operations = copy.copy(path.operations) if path is not None else []

        self.i = i
        self.j = j

        # The path cannot go any further
        if self.i < 0:
            raise ValueError('Path with a negative index')

    def finished(self):
        return self.i == 0 and self.j == 0

    def to_primitive(self):
        return [op.to_primitive() for op in reversed(self.operations)]

    def to_json(self):
        return json.dumps(self.to_primitive())

    def to_strings(self, use_colors=True):
        """Convert an edit script to a pair of strings representing the operation in a human readable way.

        :param use_colors: Boolean indicating whether to use terminal color codes to color the output.
        :return: Tuple with text corresponding to the first pronunciation and the text of the second one.
        """

        edit_script = self.to_primitive()

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
                    src_txt += u'{green}{from_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                                 op,
                                                                                                 {'width': width}))
                    dst_txt += u'{green}{to_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                               op,
                                                                                               {'width': width}))
                else:
                    src_txt += u'{red}{from_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                               op,
                                                                                               {'width': width}))
                    dst_txt += u'{red}{to_symbol: ^{width}}{normal}'.format(**_combine_dicts(colors,
                                                                                             op,
                                                                                             {'width': width}))

            elif op['op_code'] == 'insert':
                space = ' '*len(op['to_symbol'])
                src_txt += u'{on_red}{space}{normal}'.format(space=space, **_combine_dicts(colors,  op))
                dst_txt += u'{red}{to_symbol}{normal}'.format(**_combine_dicts(colors, op))

            elif op['op_code'] == 'delete':
                space = ' '*len(op['from_symbol'])
                src_txt += u'{red}{from_symbol}{normal}'.format(**_combine_dicts(colors, op))
                dst_txt += u'{on_red}{space}{normal}'.format(space=space, **_combine_dicts(colors, op))

            elif op['op_code'] == 'noninsert':
                continue

            src_txt += ' '
            dst_txt += ' '

        return src_txt, dst_txt

    def print_colors(self):
        src_txt, dst_txt = self.to_strings()
        print(u'---\n{}\n{}\n---'.format(src_txt, dst_txt))


def print_matrix(array, row_labels, col_labels, print_value=lambda x: '{}'.format(x)):
    import pandas
    printit = numpy.vectorize(print_value)
    df = pandas.DataFrame(printit(array), index=row_labels, columns=col_labels)
    print(df)


def print_state(src, trg, ops, costs):
    print_matrix(costs, src, trg)
    print('\n\n')
    print_matrix(ops, src, trg, print_value=lambda x: '({}{}{})'.format('m' if x[0] else ' ',
                                                                        'i' if x[1] else ' ',
                                                                        'd' if x[2] else ' '))


def check_insert(s, t, costs, ops, i, j, cost_method):
    _ = s
    cost = costs[i, j-1] + cost_method(t[j])
    if cost <= costs[i, j]:
        if cost < costs[i, j]:
            ops[i, j] = (False, False, False)

        costs[i, j] = cost
        ops[i, j]['insert'] = True


def check_delete(s, t, costs, ops, i, j, cost_method):
    _ = t
    cost = costs[i-1, j] + cost_method(s[i])
    if cost <= costs[i, j]:
        if cost < costs[i, j]:
            ops[i, j] = (False, False, False)

        costs[i, j] = cost
        ops[i, j]['delete'] = True


def check_match(s, t, costs, ops, i, j, cost_method):
    cost = costs[i-1, j-1] + cost_method(s[i], t[j])
    if cost <= costs[i, j]:
        if cost < costs[i, j]:
            ops[i, j] = (False, False, False)

        costs[i, j] = cost
        ops[i, j]['match'] = True

class TERMINAL(object):
    pass

def best_transforms(src, trg, op_costs=None):
    default_costs = {'match': lambda x, y: 0 if x == y else 1,
                     'insert': lambda x: 1,
                     'delete': lambda x: 1}

    op_costs = op_costs or default_costs

    insert_cost_method = op_costs['insert']
    match_cost_method = op_costs['match']
    delete_cost_method = op_costs['delete']

    src_len = len(src) + 2
    trg_len = len(trg) + 2

    # Initialize costs
    costs = numpy.ones((src_len, trg_len)) * numpy.inf

    op_type = numpy.dtype({'names': ['match',
                                     'insert',
                                     'delete'],
                           'formats': ['b',
                                       'b',
                                       'b']})
    ops = numpy.zeros((src_len, trg_len), dtype=op_type)

    #s = numpy.r_[[''], numpy.array(src), ['']]
    #t = numpy.r_[[''], numpy.array(trg), ['']]
    s = [None] + src + [None]
    t = [None] + trg + [None]

    # Set the cost of the initial cell
    costs[0, 0] = 0

    # Compute costs of inserting up to the nth target item before source
    i = 0
    for j in xrange(1, costs.shape[1]-1):
        check_insert(s, t, costs, ops, i, j, insert_cost_method)

    # Compute costs of deleting up to the nth source item before target
    j = 0
    for i in xrange(1, costs.shape[0]-1):
        check_delete(s, t, costs, ops, i, j, delete_cost_method)

    # Compute costs of everything else
    for i in xrange(1, costs.shape[0]-1):
        for j in xrange(1, costs.shape[1]-1):
            # Replace cost
            check_match(s, t, costs, ops, i, j, match_cost_method)

            # Delete cost
            check_delete(s, t, costs, ops, i, j, delete_cost_method)

            # Insert cost
            check_insert(s, t, costs, ops, i, j, insert_cost_method)

    # Compute costs of inserting up to the nth target item after the source
    i = costs.shape[0]-1
    for j in xrange(1, costs.shape[1]-1):
        check_insert(s, t, costs, ops, i, j, insert_cost_method)

    # Compute costs of deleting up to the nth source item after the target
    j = costs.shape[1]-1
    for i in xrange(1, costs.shape[0]-1):
        check_delete(s, t, costs, ops, i, j, delete_cost_method)

    # Last match cost
    # This last element does not require a cost
    # since it is matching end-of-seq to end-of-seq
    i = costs.shape[0]-1
    j = costs.shape[1]-1
    check_match(s, t, costs, ops, i, j, lambda x, y: 0)

    # Debug the state of the DP algorithm
    # print_state(s, t, ops, costs)

    # Compute transforms by backtracking paths
    paths = []

    if ops[-1, -1]['match']:
        paths.append(Script(src_len - 2, trg_len - 2))

    if ops[-1, -1]['insert']:
        paths.append(Script(src_len - 1, trg_len - 2))

    if ops[-1, -1]['delete']:
        paths.append(Script(src_len - 2, trg_len - 1))

    while any(map(lambda x: not x.finished(), paths)):
        new_paths = []

        for path in paths:
            # If we are in the top left corner of the cost matrix we are finished
            if path.finished():
                new_paths.append(path)
                continue

            if ops[path.i, path.j]['match']:
                op = Operation('match', (path.i-1) * 2 + 1, src[path.i-1], trg[path.j-1])
                new_path = Script(path.i - 1, path.j - 1, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

            if ops[path.i, path.j]['delete']:
                op = Operation('delete', (path.i-1) * 2 + 1, src[path.i-1], None)
                new_path = Script(path.i - 1, path.j, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

            if ops[path.i, path.j]['insert']:
                op = Operation('insert', (path.i-1) * 2 + 2, None, trg[path.j-1])
                new_path = Script(path.i, path.j - 1, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

        paths = new_paths

    return costs[i, j], paths, costs, ops


def main():
    a = u'm u ch a'
    b = u'm u ch o s'

    a = u'k w i ð̞ a'
    b = u'i ð̞ a'

    a = u'x e s t o'
    b = u'p ɾ i n θ e s a'

    a = u'p e k ˈe ɲ o s'
    b = u'p e k ˈe ɲ a s'

    a = u'k ˈo ð o'
    b = u'θ ˌe  θ e ð ˌi ʝ a  d ˌe ˌɛ f e θ ˌe  θ e ð ˈi ʝ a'

    if len(sys.argv) > 2:
        a = sys.argv[1]
        b = sys.argv[2]

    src = a.split()
    trg = b.split()

    distance, transfs, costs, ops = best_transforms(src, trg)


    print('distance: {}'.format(distance))
    print('costs:\n{}'.format(costs))
    print('ops:\n{}'.format(ops))
    print(transfs[0].to_json())

    for transf in transfs:
        transf.print_colors()

    print(json.dumps(transfs[0].to_primitive(), indent=4))
    transfs[0].print_colors()

if __name__ == '__main__':
    main()