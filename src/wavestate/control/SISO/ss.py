#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2022 California Institute of Technology.
# SPDX-FileCopyrightText: © 2022 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
Functions to create a SISO state space system from inputs.
"""
import numbers
import numpy as np
import warnings

from ..statespace.dense import xfer_algorithms
from ..statespace.dense import zpk_algorithms
from ..statespace.dense import ss_algorithms
from ..statespace import ssprint

from . import siso
from . import zpk


class NumericalWarning(UserWarning):
    pass


class SISOStateSpace(siso.SISO):
    """
    class to represent SISO Transfer functions using dense state space matrix representations.
    """
    def __init__(
        self,
        A, B, C, D, E,
        hermitian: bool = True,
        time_symm: bool = False,
        dt=None,
        fiducial_s=None,
        fiducial_f=None,
        fiducial_w=None,
        fiducial=None,
        fiducial_rtol=1e-4,
    ):
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)
        if E is not None:
            E = np.asarray(E)

        if hermitian:
            assert(np.all(A.imag == 0))
            assert(np.all(B.imag == 0))
            assert(np.all(C.imag == 0))
            assert(np.all(D.imag == 0))
            if E is not None:
                assert(np.all(E.imag == 0))

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.hermitian = hermitian
        self.time_symm = time_symm
        self.dt = dt

        domain_w = None
        if fiducial_f is not None:
            domain_w = 2 * np.pi * np.asarray(fiducial_f)
        if fiducial_w is not None:
            assert(domain_w is None)
            domain_w = np.asarray(fiducial_w)
        if fiducial_s is not None:
            assert(domain_w is None)
            domain_w = np.asarray(fiducial_s) / 1j

        self.test_response(
            s=fiducial_s,
            f=fiducial_f,
            w=fiducial_w,
            response=fiducial,
            rtol=fiducial_rtol,
            update=True,
        )
        return

    def test_response(
        self,
        s=None,
        f=None,
        w=None,
        response=None,
        rtol=None,
        update=False,
    ):
        domain_w = None
        if f is not None:
            domain_w = 2 * np.pi * np.asarray(f)
        if w is not None:
            assert(domain_w is None)
            domain_w = np.asarray(w)
        if s is not None:
            assert(domain_w is None)
            domain_w = np.asarray(s) / 1j

        if domain_w is not None and len(domain_w) == 0:
            if update:
                self.fiducial = domain_w
                self.fiducial_w = domain_w
                self.fiducial_rtol = rtol
                return
            return

        if rtol is None:
            rtol = self.fiducial_rtol

        if domain_w is None:
            # create a list of poiints at each resonance and zero, as well as 1 BW away
            rt_rtol = rtol**0.5
            if self.A.shape[-1] < self.N_MAX_FID:
                z, p = self._zp

                zr = z[abs(z.imag) < 1e-10]
                zc = z[z.imag > 1e-10]
                pr = p[abs(p.imag) < 1e-10]
                pc = p[p.imag > 1e-10]

                # augment the list to include midpoints between all resonances
                domain_w = np.sort(np.concatenate([
                    zr, zc.imag, abs(zc.imag) + abs(zc.real),
                    pr, pc.imag, abs(pc.imag) + abs(pc.real),
                ])).real + rt_rtol
                # and midpoints
                domain_w = np.concatenate([domain_w, (domain_w[0:-1] + domain_w[1:])/2])
            else:
                warnings.warn("StateSpace is large (>100 states), using reduced response fiducial auditing heuristics. TODO to make this smarter", NumericalWarning)
                domain_w = np.asarray([rt_rtol])

        self_response = self.response(w=domain_w)
        if response is not None:
            if callable(response):
                response = response(w=domain_w)
            np.testing.assert_allclose(
                self_response,
                response,
                atol=0,
                rtol=rtol,
                equal_nan=False,
            )
        else:
            # give it one chance to select better points
            select_bad = (~np.isfinite(self_response)) | (self_response == 0)
            if update and np.any(select_bad):
                if np.all(select_bad):
                    domain_w = np.array([rt_rtol])
                    self_response = self.response(w=domain_w)
                else:
                    self_response = self_response[~select_bad]
                    domain_w = domain_w[~select_bad]
            response = self_response

        if update:
            self.fiducial = response
            self.fiducial_w = domain_w
            self.fiducial_rtol = rtol
        return

    @property
    def ABCDE(self):
        if self.E is None:
            E = np.eye(self.A.shape[-1])
        else:
            E = self.E
        return self.A, self.B, self.C, self.D, E

    @property
    def ABCDe(self):
        return self.A, self.B, self.C, self.D, self.E

    @property
    def ABCD(self):
        if self.E is None:
            raise RuntimeError("Cannot Drop E")
        else:
            assert(np.all(np.eye(self.E.shape[-1]) == self.E))
            self.E = None
        return self.A, self.B, self.C, self.D

    def __iter__(self):
        """
        Represent self like a typical scipy zpk tuple. This throws away symmetry information
        """
        yield self.A
        yield self.B
        yield self.C
        yield self.D
        if self.E is not None:
            yield self.E

    _zp_tup = None

    @property
    def _zp(self):
        """
        Create a raw z, p tuple from the direct calculation
        """
        if self._zp_tup is None:
            z, p = zpk_algorithms.ss2zp(
                A=self.A,
                B=self.B,
                C=self.C,
                D=self.D,
                E=self.E,
                idx_in=0,
                idx_out=0,
                fmt="scipy",
            )
            self._zp_tup = (z, p)
        return self._zp_tup

    _ZPK = None

    @property
    def asZPK(self):
        if self._ZPK is not None:
            self._ZPK
        z, p = self._zp
        # the gain is not specified here,
        # as it is established from the fiducial data
        self._ZPK = zpk.zpk(
            z, p,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            convention='scipy',
            fiducial=self.fiducial,
            fiducial_w=self.fiducial_w,
            fiducial_rtol=self.fiducial_rtol,
        )
        return self._ZPK

    @property
    def asSS(self):
        return self

    def print_nonzero(self):
        """
        """
        return ssprint.print_dense_nonzero(self)

    def response(self, f=None, w=None, s=None):
        domain = None
        if f is not None:
            domain = 2j * np.pi * np.asarray(f)
        if w is not None:
            assert(domain is None)
            domain = 1j * np.asarray(w)
        if s is not None:
            assert(domain is None)
            domain = np.asarray(s)

        return xfer_algorithms.ss2response_siso(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            s=domain,
            idx_in=0,
            idx_out=0,
        )

    def inv(self):
        ABCDE = ss_algorithms.inverse_DSS(*self.ABCDE)
        return self.__class__(
            A=ABCDE.A,
            B=ABCDE.B,
            C=ABCDE.C,
            D=ABCDE.D,
            E=ABCDE.E,
            hermitian=self.hermitian,
            time_symm=self.time_symm,
            dt=self.dt,
            fiducial=1/self.fiducial,
            fiducial_w=self.fiducial_w,
            fiducial_rtol=self.fiducial_rtol,
        )

    def __mul__(self, other):
        """
        """
        if isinstance(other, siso.SISO):
            other = other.asSS
            hermitian = self.hermitian and other.hermitian
            time_symm = self.time_symm and other.time_symm
            assert(self.dt == other.dt)
            ABCDE = ss_algorithms.chain([self.ABCDE, other.ABCDE])

            if len(self.fiducial_w) + len(other.fiducial_w) < self.N_MAX_FID:
                slc = slice(None, None, 1)
            else:
                slc = slice(None, None, 2)
            fid_other_self = other.response(w=self.fiducial_w[slc])
            fid_self_other = self.response(w=other.fiducial_w[slc])
            assert(self.dt == other.dt)
            return self.__class__(
                A=ABCDE.A,
                B=ABCDE.B,
                C=ABCDE.C,
                D=ABCDE.D,
                E=ABCDE.E,
                hermitian=hermitian,
                time_symm=time_symm,
                dt=self.dt,
                fiducial=np.concatenate([
                    self.fiducial[slc] * fid_other_self,
                    fid_self_other * other.fiducial[slc]
                ]),
                fiducial_w=np.concatenate([
                    self.fiducial_w[slc],
                    other.fiducial_w[slc]
                ]),
                fiducial_rtol=self.fiducial_rtol,
            )
        elif isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B * other,
                C=self.C,
                D=self.D * other,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=self.fiducial * other,
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=other * self.C,
                D=other * self.D,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=other * self.fiducial,
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            return self.__class__(
                A=self.A,
                B=self.B,
                C=self.C / other,
                D=self.D / other,
                E=self.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=self.fiducial / other,
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
            )
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            ABCDE = ss_algorithms.inverse_DSS(*self.ABCDE)
            return self.__class__(
                A=ABCDE.A,
                B=ABCDE.B,
                C=other * ABCDE.C,
                D=other * ABCDE.D,
                E=ABCDE.E,
                hermitian=self.hermitian,
                time_symm=self.time_symm,
                dt=self.dt,
                fiducial=other / self.fiducial,
                fiducial_w=self.fiducial_w,
                fiducial_rtol=self.fiducial_rtol,
            )
        else:
            return NotImplemented

    def __pow__(self, other):
        """
        """
        if isinstance(other, numbers.Number):
            if other == -1:
                return self.inv()
            elif other == 1:
                return self
            else:
                return NotImplemented
        else:
            return NotImplemented

def ss(
    *args,
    hermitian=True,
    time_symm=False,
    dt=None,
    fiducial=None,
    fiducial_w=None,
    fiducial_f=None,
    fiducial_s=None,
    fiducial_rtol=1e-6,
):
    """
    Form a SISO LTI system from statespace matrices.

    """
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, siso.SISO):
            arg = arg.asSS
        if isinstance(arg, SISOStateSpace):
            return arg
        elif isinstance(arg, (tuple, list)):
            A, B, C, D, E = arg
    elif len(args) == 4:
        A, B, C, D = args
        E = None
    elif len(args) == 5:
        A, B, C, D, E = args
    else:
        raise RuntimeError("Unrecognized argument format")
    return SISOStateSpace(
        A, B, C, D, E,
        hermitian=True,
        time_symm=False,
        dt=None,
        fiducial=fiducial,
        fiducial_s=fiducial_s,
        fiducial_f=fiducial_f,
        fiducial_w=fiducial_w,
        fiducial_rtol=fiducial_rtol,
    )


