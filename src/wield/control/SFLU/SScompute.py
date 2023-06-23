#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Massachusetts Institute of Technology.
# SPDX-FileCopyrightText: © 2021 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import numpy as np
import yaml
import re

from collections import defaultdict

from wield.bunch import Bunch

from wield.utilities.np import matrix_stack

from ..import string_tuple_keys as stk
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import sympy as sp
except ImportError:
    sp = None

from . import SFLU
from .conversions import (
    yamlstr_convert,
    yamlstr_convert_rev,
    normalize_list2tuple,
)

from .. import ss_bare
from .. import MIMO


class SSCompute:
    """
    An SFLU computation class designed to use statespaces for the computation.

    This should perhaps live somewhere else. Here is an acceptable home for the moment
    """
    def __init__(
        self,
        edges,
        row2col,
        col2row,
        typemap_to=None,
        eye=1,
    ):
        self.edges = edges
        self.row2col = row2col
        self.col2row = col2row

        edges_rev = defaultdict(set)

        for edge, v in self.edges:
            edges_rev[v].add(edge)

        self.edges_rev = dict(edges_rev)

        if typemap_to is None:
            typemap_to = self.typemap_to_default
        self.typemap_to = typemap_to
        self.eye = self.typemap_to(eye)
        return

    def typemap_to_default(self, v):
        """
        Default conversion for edge and node values.
        This conversion promotes scalars and arrays to become
        1x1 matrices so that the matmul operation may be applied
        """
        if isinstance(v, list):
            v = matrix_stack(v)
        elif isinstance(v, ss_bare.ss.BareStateSpace):
            return v

        v = np.asarray(v)
        if len(v.shape) == 0:
            v = v * self.eye
        else:
            assert (len(v.shape) == 2)
            #
            v = ss_bare.ss.BareStateSpace.fromD(v)
        return v

    def edge_map(self, edge_map, default=False):
        """
        Map the edges into an edge space for computation.
        """
        def edge_compute(ev):
            if isinstance(ev, str):
                # if the value is a string, then we pull it out of the edge map

                # first check if we should
                # perform some mapping operations on the string to evaluate
                # simple mathematical operations like unary negation
                if ev[0] == '-':
                    ev = ev[1:]
                    if default is not False:
                        return self.typemap_to(-edge_map.setdefault(ev, default))
                    else:
                        return self.typemap_to(-edge_map[ev])
                else:
                    # no operation, so directly compute the mapping
                    if default is not False:
                        return self.typemap_to(edge_map.setdefault(ev, default))
                    else:
                        return self.typemap_to(edge_map[ev])
            else:
                # then it must be an edge computation
                op = ev[0]
                if op == '*':
                    prod = edge_compute(ev[1])
                    for v in ev[2:]:
                        prod = prod @ edge_compute(v)
                    return prod
                elif op == '-':
                    if len(ev) == 2:
                        return -edge_compute(ev[1])
                    elif len(ev) == 3:
                        return edge_compute(ev[1])-edge_compute(ev[2])
                    else:
                        assert(False)
                else:
                    raise NotImplementedError("Unrecognized edge computation opcode")
            return

        Espace = {}
        for ek, ev in self.edges.items():
            assert(isinstance(ek, stk.EdgeTuple))
            Espace[ek] = edge_compute(ev)

        return Espace

    def SScompletion(self, edge_map):
        Espace = self.edge_map(edge_map)
        Elist = []
        Inodes = set()
        Onodes = set()
        for (r, c), E in Espace.items():
            Onodes.add(r)
            Inodes.add(c)
            # make a statespace with just a single named input and output from the entire set
            # use the node names for the edges as the IO kw name
            ss = MIMO.MIMOStateSpace(
                ss=E,
                inputs={c: (0, E.Ninputs)},
                outputs={r: (0, E.Noutputs)},
            )
            Elist.append(ss)

        ss_disco = MIMO.ssjoinsum(*Elist)

        # TODO, probably need to rescale here

        # make a connection from every output to every input
        # this is the equivalence of the signal flow graph method
        # it should only be done on the intersection of the outputs and inputs
        connections = [(n, n) for n in (Inodes & Onodes)]
        ss_conn = ss_disco.feedback_connect(connections=connections)
        ss_conn = ss_conn.balance()

        # TODO, probably need to rescale and convert to Schur form here

        self.ss = ss_conn
        self.ss_disconnected = ss_disco
        self.connections = connections

        return Bunch(
            ss=ss_conn,
            ss_disconnected=ss_disco,
            connections=connections,
        )

    def inverse_col_single_fresponse(self, Rset, C, F_Hz, derivatives=False):
        """
        Find the inverse of a single column into many rows given by Rset

        derivatives: if True, include all derivative testpoints in Rset
        """
        return self.inverse_col_fresponse(Rset, {C: None}, F_Hz=F_Hz, derivatives=derivatives)

    def inverse_col_fresponse(self, Rset, Cmap, F_Hz, derivatives=False):
        """
        This computes the matrix element for an inverse from C to R.

        The columns are a dictionary mapping column names to values. The values can be vectors, matrices or None.
        A value of None implicitly chooses the identity matrix.
        """
        if derivatives:
            raise NotImplementedError()
        Rset = set(stk.key_map(R) for R in Rset)
        Cmap = {stk.key_map(C): v for C, v in Cmap.items()}

        plantSS = self.ss[
            list(Rset),
            list(Cmap.keys())
        ].minreal(scale_io=True, tol=1e-10)

        fr = plantSS.fresponse(f=F_Hz)
        resultsAC = dict()

        for r, val in Rset:
            val_sum = None
            for c in Cmap.items():
                if val is None:
                    val = fr[[r], [c]].tf
                else:
                    val = fr[[r], [c]].tf @ val
                if val_sum is None:
                    val_sum = val
                else:
                    # don't do this in place for better broadcasting
                    val_sum = val_sum + val
            # NOTE THE MINUS SIGN
            # needed for consistency with the definition of signal flow graphs
            resultsAC[r] = val_sum

        return resultsAC

    def inverse_row_single_fresponse(self, R, Cset, F_Hz, derivatives=False):
        """
        Find the inverse of a single row from many columns given by Cset
        """
        return self.inverse_row_fresponse({R: None}, Cset, F_Hz, derivatives=derivatives)

    def inverse_row_fresponse(self, Rmap, Cset, F_Hz, derivatives=False):
        """
        This computes the matrix element for an inverse from C to R

        derivatives: if True, include all derivative excitations in Cset
        """
        if derivatives:
            raise NotImplementedError()
        Cset = set(stk.key_map(C) for C in Cset)
        Rmap = {stk.key_map(R): v for R, v in Rmap.items()}

        plantSS = self.ss[
            list(Rmap.keys()),
            list(Cset)
        ]
        fr = plantSS.fresponse(f=F_Hz)
        resultsAC = dict()

        for c in Cset:
            val_sum = None
            for r, val in Rmap.items():
                if val is None:
                    val = fr[[r], [c]].tf
                else:
                    val = fr[[r], [c]].tf @ val
                if val_sum is None:
                    val_sum = val
                else:
                    # don't do this in place for better broadcasting
                    val_sum = val_sum + val
            # NOTE THE MINUS SIGN
            # needed for consistency with the definition of signal flow graphs
            resultsAC[c] = val_sum

        return resultsAC

