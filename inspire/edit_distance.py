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
            pass

    def to_primitive(self):
        return [op.to_primitive() for op in reversed(self.operations)]

    def to_json(self):
        return json.dumps(self.to_primitive())

    def finished(self):
        return self.i < 0 and self.j < 0

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

    def __repr__(self):
        return u'{}'.format([u'{}'.format(op) for op in self.operations[::-1] if not op.from_symbol == op.to_symbol])


def best_transforms(src, trg, op_costs=None):
    default_costs = {'match': lambda x, y: 0 if x == y else 1,
                     'insert': lambda x: 1,
                     'delete': lambda x: 1}

    op_costs = op_costs or default_costs

    INSERT = op_costs['insert']
    MATCH = op_costs['match']
    DELETE = op_costs['delete']

    src_len = len(src)
    trg_len = len(trg)

    # Initialize costs
    costs = numpy.ones((src_len, trg_len)) * numpy.inf

    op_type = numpy.dtype({'names': ['match',
                                     'insert',
                                     'delete'],
                           'formats': ['b',
                                       'b',
                                       'b']})
    ops = numpy.zeros((src_len, trg_len), dtype=op_type)

    s = numpy.array(src)
    t = numpy.array(trg)

    # Compute cost of 1st source item and 1st target item
    costs[0, 0] = MATCH(s[0], t[0])
    ops[0, 0]['match'] = True
    i, j = 0, 0

    # Compute costs of 1st source item and nth target item
    for j in xrange(1, costs.shape[1]):
        costs[0, j] = sum([INSERT(t[k]) for k in xrange(j)]) + MATCH(s[0], t[j])
        ops[0, j]['match'] = True

    # Compute costs of 1st target item and nth source item
    for i in xrange(1, costs.shape[0]):
        costs[i, 0] = sum([DELETE(s[k]) for k in xrange(i)]) + MATCH(s[i], t[0])
        ops[i, 0]['match'] = True

    # Compute costs of everything else
    for i in xrange(1, costs.shape[0]):
        for j in xrange(1, costs.shape[1]):
            # Replace cost
            cost = costs[i - 1, j - 1] + MATCH(s[i], t[j])
            if cost <= costs[i, j]:
                if cost < costs[i, j]:
                    ops[i, j] = (False, False, False)

                costs[i, j] = cost
                ops[i, j]['match'] = True

            # Delete cost
            cost = costs[i - 1, j] + DELETE(s[i])
            if cost <= costs[i, j]:
                if cost < costs[i, j]:
                    ops[i, j] = (False, False, False)

                costs[i, j] = cost
                ops[i, j]['delete'] = True

            # Insert cost
            cost = costs[i, j - 1] + INSERT(t[j])
            if cost <= costs[i, j]:
                if cost < costs[i, j]:
                    ops[i, j] = (False, False, False)

                costs[i, j] = cost
                ops[i, j]['insert'] = True

    # Compute transforms by backtracking paths
    paths = [Script(src_len - 1, trg_len - 1)]
    while not all(map(lambda x: x.finished(), paths)):
        new_paths = []

        for path in paths:
            # Check if we have aligned up to the begining of target or source
            if path.i < 0 or path.j < 0:
                # If we are in the top left corner of the cost matrix we are finished
                if path.finished():
                    new_paths.append(path)
                    continue

                # Else we have to add the insertions or deletions required to reach it
                else:
                    for k in xrange(path.i, -1, -1):
                        op = Operation('delete', k * 2 + 1, src[k], None)
                        path.operations.append(op)
                        path.i = k - 1

                    for k in xrange(path.j, -1, -1):
                        op = Operation('insert', 0, None, trg[k])
                        path.operations.append(op)
                        path.j = k - 1

                    new_paths.append(path)
                    continue

            if ops[path.i, path.j]['match']:
                op = Operation('match', path.i * 2 + 1, src[path.i], trg[path.j])
                new_path = Script(path.i - 1, path.j - 1, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

            if ops[path.i, path.j]['delete']:
                op = Operation('delete', path.i * 2 + 1, src[path.i], None)
                new_path = Script(path.i - 1, path.j, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

            if ops[path.i, path.j]['insert']:
                op = Operation('insert', path.i * 2 + 2, None, trg[path.j])
                new_path = Script(path.i, path.j - 1, path=path)
                new_path.operations.append(op)
                new_paths.append(new_path)

        paths = new_paths

    return costs[i, j], paths, costs, ops


if __name__ == '__main__':
    a = u'm u ch a'
    b = u'm u ch o s'

    a = u'k w i ð̞ a'
    b = u'i ð̞ a'

    a = u'x e s t o'
    b = u'p ɾ i n θ e s a'

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
    for transf in transfs: transf.print_colors()
