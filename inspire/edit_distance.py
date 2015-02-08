#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import sys
import json

import numpy


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

    def to_primitive(self):
        return [op.to_primitive() for op in reversed(self.operations)]

    def to_json(self):
        return json.dumps(self.to_primitive())

    def finished(self):
        return self.i == 0 and self.j == 0

    def print_colors(self):
        from blessings import Terminal

        term = Terminal()

        src_txt = u''
        dst_txt = u''
        for op in reversed(self.operations):
            if op.op_code == 'match':
                if op.from_symbol == op.to_symbol:
                    src_txt += u'{t.green}{op.from_symbol}{t.normal}'.format(t=term, op=op)
                    dst_txt += u'{t.green}{op.to_symbol}{t.normal}'.format(t=term, op=op)
                else:
                    src_txt += u'{t.red}{op.from_symbol}{t.normal}'.format(t=term, op=op)
                    dst_txt += u'{t.red}{op.to_symbol}{t.normal}'.format(t=term, op=op)

            elif op.op_code == 'insert':
                src_txt += u'{t.on_red} {t.normal}'.format(t=term, op=op)
                dst_txt += u'{t.red}{op.to_symbol}{t.normal}'.format(t=term, op=op)

            elif op.op_code == 'delete':
                src_txt += u'{t.red}{op.from_symbol}{t.normal}'.format(t=term, op=op)
                dst_txt += u'{t.on_red} {t.normal}'.format(t=term, op=op)

            src_txt += u' '
            dst_txt += u' '

        print(u'---\n{}\n{}\n---'.format(src_txt, dst_txt))

    #def __repr__(self):
    #    return u'{}'.format([u'{}'.format(op) for op in self.operations[::-1] if not op.from_symbol == op.to_symbol])


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

    s = numpy.r_[[''], numpy.array(src), ['']]
    t = numpy.r_[[''], numpy.array(trg), ['']]

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
    for j in xrange(1, costs.shape[1]):
        check_insert(s, t, costs, ops, i, j, insert_cost_method)

    # Compute costs of deleting up to the nth source item after the target
    j = costs.shape[1]-1
    for i in xrange(1, costs.shape[0]):
        check_delete(s, t, costs, ops, i, j, delete_cost_method)

    # Last match cost
    i = costs.shape[0]-1
    j = costs.shape[1]-1
    check_match(s, t, costs, ops, i, j, match_cost_method)

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


if __name__ == '__main__':
    main()