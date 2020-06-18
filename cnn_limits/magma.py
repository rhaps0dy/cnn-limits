import atexit
import collections
import ctypes
import os
from ctypes import (CDLL, POINTER, byref, c_float, c_int, c_int64, c_longlong,
                    c_uint)

import numpy as np
import torch
from numpy.ctypeslib import as_array, as_ctypes, ndpointer

__all__ = ['posv', 'potrf', 'potri', 'make_symm', 'syevd', 'EigenOut']

# Load the library
# libmagma_path = "/usr/local/magma/lib/libmagma.so"
libmagma_path = os.path.join(os.environ['HOME'], 'magma/lib/libmagma.so')
try:
    libmagma = CDLL(libmagma_path)
except OSError as e:
    raise OSError(("{}. Please edit `libmagma_path` in \"{}\" to the correct "
                   "location".format(e, __file__)))


# MAGMA dtypes
enum = c_uint
class vec_t:
    MagmaNoVec = 301
    MagmaVec   = 302
magma_int = c_longlong  # because we linked with mkl_intel_ilp64, instead of mkl_intel_lp64
class uplo:
    upper         = 121
    lower         = 122
    full          = 123  # lascl, laset
    hessenberg    = 124  # lascl

    safe_mode = False

    @classmethod
    def transpose(klass, uplo_A):
        if uplo_A == klass.lower:
            return klass.upper
        elif uplo_A == klass.upper:
            return klass.lower
        raise ValueError(f"uplo_A not lower or upper: {uplo_A}")

    @classmethod
    def check(klass, M, t):
        if klass.safe_mode:
            M = np.reshape(M, (-1,) + M.shape[-2:])
            if t == klass.upper:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        assert not np.any(np.isnan(M[i, j, j:])), (
                            "M is not upper triangular")
            elif t == klass.lower:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        assert not np.any(np.isnan(M[i, j, :j+1])), (
                            "M is not lower triangular")
            elif t == klass.full:
                assert not np.any(np.isnan(M)), (
                            "M is not full")


    @classmethod
    def enforce(klass, M, t):
        if klass.safe_mode:
            M = np.reshape(M, (-1,) + M.shape[-2:])
            if t == klass.upper:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        M[i, j, :j] = np.nan
            elif t == klass.lower:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        M[i, j, j+1:] = np.nan


# Functions
def dtype_iscuda(A):
    if isinstance(A, np.ndarray):
        return (A.dtype, False)
    elif isinstance(A, torch.Tensor):
        return (A.dtype, A.is_cuda)
    else:
        raise Exception()

def lda(A):
    try:  # numpy
        return A.strides[-1] // A.dtype.itemsize
    except AttributeError: # torch
        return A.stride()[-1]

by_dtype_posv = {
    (np.dtype(np.float32), False): libmagma.magma_sposv,
    (np.dtype(np.float64), False): libmagma.magma_dposv,
    (torch.float32, True): libmagma.magma_sposv_gpu,
    (torch.float32, False): libmagma.magma_sposv,
}
for dtype, f in by_dtype_posv.items():
    if dtype[0] is torch.float32:
        A_type = POINTER(c_float)
    else:
        A_type = ndpointer(dtype[0], ndim=2, flags=('F', 'ALIGNED', 'WRITEABLE'))
    f.argtypes = [
        enum,                          # uplo
        magma_int, magma_int,          # n, nrhs
        A_type,                        # A
        magma_int,                     # lda
        A_type,                        # b
        magma_int,                     # ldb
        POINTER(magma_int),            # info
    ]
    f.restype = magma_int  # info
    f.argnames = ["uplo", "n", "nrhs", "A", "lda", "B", "ldb", "info"]
def posv(A, b, lower=False):
    """
    Solve linear system `Ax = b`, using the Cholesky factorisation of `A`.
    `A` must be positive definite.
    `A` and `b` will be overwritten during the routine, with the Cholesky
    factorisation and the solution respectively.
    Only the strictly (lower/upper) triangle of `A` is accessed.
    """
    info = magma_int()
    if not (A.shape[0] == A.shape[1]
            and A.shape[1] == b.shape[0]):
        raise ValueError("Bad shape of A, b ({}, {})".format(A.shape, b.shape))
    uplo_A = uplo.lower if lower else uplo.upper
    uplo.check(A, uplo_A)

    if isinstance(A, np.ndarray):
        assert isinstance(b, np.ndarray), "A and B have to be of the same type"
        if not A.flags['F']:
            A = A.T  # make C array into Fortran array
            uplo_A = uplo.transpose(uplo_A)
        if not b.flags['F']:
            b = b.T.copy().T
        A_ptr, b_ptr = A, b
    elif isinstance(A, torch.Tensor):
        if not A.t().is_contiguous():
            A = A.t()
            uplo_A = uplo.transpose(uplo_A)
        if not b.t().is_contiguous():
            b = b.t().clone().t()
        assert A.t().is_contiguous(), "`A` is not in Fortran format"
        #assert 'cuda' in A.device.type, "`A` is not in GPU, use Numpy version."
        assert b.t().is_contiguous(), "`b` is not in Fortran format"
        #assert 'cuda' in b.device.type, "`b` is not in GPU, use Numpy version."
        assert A.is_cuda == b.is_cuda
        A_ptr = ctypes.cast(A.data_ptr(), POINTER(c_float))
        b_ptr = ctypes.cast(b.data_ptr(), POINTER(c_float))

    args = [uplo_A,
            A.shape[0], b.shape[1],
            A_ptr, lda(A),
            b_ptr, lda(b)]
    f = by_dtype_posv[dtype_iscuda(A)]
    f(*args, byref(info))

    info = info.value
    if info < 0:
        if info == -113:
            raise RuntimeError("MAGMA could not malloc GPU device memory")
        raise ValueError("Illegal {}th argument: {} = {}"
                         .format(-info, f.argnames[-info], args[-info]))
        # it could also be that the error is a MAGMA_ERR definition, see
        # `magma_types.h`
    if info > 0:
        raise np.linalg.LinAlgError(
            f"The order-{info} leading minor of A is not positive "
            "definite, so the factorization could not be completed")

    uplo.enforce(A, uplo_A)
    return b


by_dtype_potrf = {
    np.dtype(np.float32): libmagma.magma_spotrf_m,
    np.dtype(np.float64): libmagma.magma_dpotrf_m,
    torch.float32: libmagma.magma_spotrf_gpu,
}
for dtype in [np.dtype(np.float32), np.dtype(np.float64)]:
    f = by_dtype_potrf[dtype]
    f.argtypes = [
        magma_int,                     # ngpu
        enum,                          # uplo
        magma_int,                     # n
        ndpointer(dtype, ndim=2, flags=('F', 'ALIGNED', 'WRITEABLE')), # A
        magma_int,                     # lda
        POINTER(magma_int),            # info
    ]
    f.restype = magma_int  # info
    f.argnames = ["ngpu", "uplo", "n", "A", "lda", "info"]

for dtype in [torch.float32]:
    f = by_dtype_potrf[dtype]
    f.argtypes = [
        enum,                          # uplo
        magma_int,                     # n
        POINTER(c_float),              # dA
        magma_int,                     # ldda
        POINTER(magma_int),            # info
    ]
    f.restype = magma_int  # info
    f.argnames = ["uplo", "n", "dA", "ldda", "info"]


def potrf(A, lower=False, n_gpu=1):
    """
    Compute the Cholesky factorisation of positive definite matrix `A`.
    `A` will be overwritten during the routine with its Cholesky factorisation.
    Only the strictly (lower/upper) triangle of `A` is accessed.
    """
    info = magma_int()
    if not (A.shape[0] == A.shape[1]):
        raise ValueError("Matrix A must be square, is {}.".format(A.shape))
    uplo_A = uplo.lower if lower else uplo.upper
    uplo.check(A, uplo_A)

    if isinstance(A, np.ndarray):
        if not A.flags['F']:
            A = A.T
            uplo_A = uplo.transpose(uplo_A)
        args = [n_gpu,
                uplo_A,
                A.shape[0],
                A, lda(A)]
    elif isinstance(A, torch.Tensor):
        assert A.t().is_contiguous(), "`A` is not in Fortran format"
        assert 'cuda' in A.device.type, "`A` is not in GPU, use Numpy version."
        args = [uplo_A,
                A.shape[0],
                ctypes.cast(A.data_ptr(), POINTER(c_float)),
                lda(A)]
    else:
        raise ValueError("type of A: {}".format(type(A)))
    f = by_dtype_potrf[A.dtype]
    f(*args, byref(info))

    info = info.value
    if info < 0:
        if info == -113:
            raise RuntimeError("MAGMA could not malloc GPU device memory")
        raise ValueError("Illegal {}th argument: {} = {}"
                         .format(-info, f.argnames[-info], args[-info]))
        # it could also be that the error is a MAGMA_ERR definition, see
        # `magma_types.h`
    if info > 0:
        raise np.linalg.LinAlgError(
            f"The order-{info} leading minor of A is not positive "
            "definite, so the factorization could not be completed")
    uplo.enforce(A, uplo_A)
    return A


by_dtype_potri = {
    np.dtype(np.float32): libmagma.magma_spotri,
    np.dtype(np.float64): libmagma.magma_dpotri,
}
for dtype, f in by_dtype_potri.items():
    f.argtypes = [
        enum,                          # uplo
        magma_int,                     # n
        ndpointer(dtype, ndim=2, flags=('F', 'ALIGNED', 'WRITEABLE')), # A
        magma_int,                     # lda
        POINTER(magma_int),            # info
    ]
    f.restype = magma_int  # info
def potri(A, lower=False):
    """
    Compute the inverse of triangular matrix `A`.
    `A` will be overwritten during the routine with its inverse factorisation.
    Only the strictly (lower/upper) triangle of `A` is accessed.
    """
    info = magma_int()
    if not (A.shape[0] == A.shape[1]):
        raise ValueError("Matrix A must be square, is {}.".format(A.shape))
    uplo_A = uplo.lower if lower else uplo.upper
    uplo.check(A, uplo_A)

    argnames = ["uplo", "n", "A", "lda", "info"]
    args = [uplo_A,
            A.shape[0],
            A, lda(A),
            byref(info)]
    by_dtype_potri[A.dtype](*args)
    uplo.enforce(A, uplo_A)

    info = info.value
    if info < 0:
        if info == -113:
            raise RuntimeError("MAGMA could not malloc GPU device memory")
        raise ValueError("Illegal {}th argument: {} = {}"
                         .format(-info, argnames[-info], args[-info]))
        # it could also be that the error is a MAGMA_ERR definition, see
        # `magma_types.h`
    if info > 0:
        raise ValueError("Error code: {}".format(info))
    uplo.enforce(A, uplo_A)
    return A


EigenOut = collections.namedtuple("EigenOut", ("vals", "vecs"))


by_dtype_syevd = {
    np.dtype(np.float32): libmagma.magma_ssyevd,
    np.dtype(np.float64): libmagma.magma_dsyevd,
}
for dtype in [np.dtype(np.float32), np.dtype(np.float64)]:
    f = by_dtype_syevd[dtype]
    f.argtypes = [
        enum,                          # jobz
        enum,                          # uplo
        magma_int,                     # n
        ndpointer(dtype, ndim=2, flags=('F', 'ALIGNED', 'WRITEABLE')), # A
        magma_int,                     # lda
        ndpointer(dtype, ndim=1, flags=('F', 'ALIGNED', 'WRITEABLE')), # w
        ndpointer(dtype, ndim=1, flags=('F', 'ALIGNED', 'WRITEABLE')), # work
        magma_int,                     # lwork
        ndpointer(np.int64, ndim=1, flags=('F', 'ALIGNED', 'WRITEABLE')), # iwork
        magma_int,                     # liwork
        POINTER(magma_int),            # info
    ]
    f.restype = magma_int  # info
    f.argnames = ["jobz", "uplo", "n", "A", "lda", "w", "work",
                  "lwork", "iwork", "liwork", "info"]
def syevd(A, vectors=False, lower=True):
    "computes eigenvalues and optionally eigenvectors of PSD matrix A"
    info = magma_int()
    if not (A.shape[0] == A.shape[1]):
        raise ValueError("Matrix A must be square, is {}.".format(A.shape))
    jobz = vec_t.MagmaVec if vectors else vec_t.MagmaNoVec
    uplo_A = uplo.lower if lower else uplo.upper
    uplo.check(A, uplo_A)

    f = by_dtype_syevd[A.dtype]
    if isinstance(A, np.ndarray):
        if not A.flags['F']:
            A = A.T
            uplo_A = uplo.transpose(uplo_A)
        w = np.empty(A.shape[0], A.dtype, order='F')
        work = np.empty(1, A.dtype, order='F')
        iwork = np.empty(1, np.int64, order='F')
        args = [jobz,
                uplo_A,
                A.shape[0], A, lda(A),
                w,
                work, -1,
                iwork, -1]
        f(*args, byref(info))
        assert info.value == 0, "something is wrong"

        lwork = int(work[0])
        liwork = int(iwork[0])
        print(f"lwork={lwork}, liwork={liwork}, N={A.shape[0]}")

        work = np.empty(lwork, A.dtype, order='F')
        iwork = np.empty(liwork, np.int64, order='F')
        args = [jobz,
                uplo_A,
                A.shape[0], A, lda(A),
                w,
                work, lwork,
                iwork, liwork]
    elif isinstance(A, torch.Tensor):
        raise NotImplementedError
        assert A.t().is_contiguous(), "`A` is not in Fortran format"
        assert 'cuda' in A.device.type, "`A` is not in GPU, use Numpy version."
    else:
        raise ValueError("type of A: {}".format(type(A)))
    f(*args, byref(info))

    info = info.value
    if info < 0:
        if info == -113:
            raise RuntimeError("MAGMA could not malloc GPU device memory")
        raise ValueError("Illegal {}th argument: {} = {}"
                         .format(-info, f.argnames[-info], args[-info]))
        # it could also be that the error is a MAGMA_ERR definition, see
        # `magma_types.h`
    if info > 0:
        raise np.linalg.LinAlgError("Failed to converge")
    uplo.enforce(A, uplo_A)
    return EigenOut(w, A)


def make_symm(A, lower=False):
    i_lower = np.tril_indices(A.shape[-1], -1)
    if lower:
        A.T[i_lower] = A[i_lower]
    else:
        A[i_lower] = A.T[i_lower]
    return A


def poi(A):
    pre_uplo = uplo.safe_mode
    try:
        uplo.safe_mode = False
        A = potri(potrf(A))
    finally:
        uplo.safe_mode = pre_uplo
    make_symm(A)
    return A


libmagma.magma_init.argtypes = []
libmagma.magma_init.restype = magma_int
def init():
    info = libmagma.magma_init()
    if info != 0:
        raise ValueError("MAGMA `init` returned {}".format(info))


libmagma.magma_finalize.argtypes = []
libmagma.magma_finalize.restype = magma_int
def finalize():
    info = libmagma.magma_finalize()
    if info != 0:
        raise ValueError("MAGMA `finalize` returned {}".format(info))


init()
atexit.register(finalize)


if __name__ == '__main__':
    A = np.random.randn(4, 4)
    A = np.array(A @ A.T, order='F', dtype=np.float64)
    b = np.array([1,2,3,4])[:, None]
    b = np.array(b, order='F', dtype=np.float64)

    print("Ax = b")
    print("A=", A)
    print("b=", b)

    np_out = np.linalg.solve(A, b)
    b_old = np.copy(b)
    A_old = np.copy(A)
    print(np_out, A, A @ np_out, b)

    magma_out = posv(A, b)
    print(magma_out, A_old @ magma_out, b_old)

    A = A_old
    L = np.linalg.cholesky(A)
    potrf(A, lower=True)
    print(L, np.tril(A))


if __name__ == '__main__':
    A = np.random.randn(10, 10)
    A = np.array(A.T @ A, order='F', dtype=np.float32)
    A_old = np.copy(A)

    print("choleskying...")
    potrf(A, lower=True)
    print("done!")

    L = np.tril(A)
    assert np.allclose(L@L.T, A_old)

if __name__ == '__main__':
    A = torch.randn((10, 10), device='cuda')
    # A = np.array(A.T @ A, order='F', dtype=np.float32)
    A = (A.t() @ A).t()
    A_old = A.clone()

    print("choleskying...")
    potrf(A, lower=True)
    print("done!")
    L = torch.tril(A)
    assert np.allclose((L@L.t()).cpu().numpy(), A_old.cpu().numpy(), rtol=1e-5)
