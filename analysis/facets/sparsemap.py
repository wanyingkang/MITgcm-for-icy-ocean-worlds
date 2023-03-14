from warnings import warn
from functools import reduce
import operator
import re
import os
import glob
import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix

__all__ = ['SparseMap', 'savemap']

uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def indices(s, len):
    try:
        start, stop, step = s.indices(len)
    except AttributeError:
        start = np.clip(s if s >= 0 else len + s, 0, len)
        stop = np.clip(start + 1, 0, len)
        step = 1
    return start, stop, step

def savemap(basename, m, shape, isize=None, formats=['i4','i4','f8']):
    coo = m.tocoo()
    a = np.rec.fromarrays([coo.row, coo.col, coo.data], formats=formats, names=['i','j','w'])
    fmts = ''.join(formats)
    shapes = 'x'.join(str(i) for i in shape[::-1])
    if isize is None:
        isize = m.shape[1]
    a.tofile('{}.{}_{}x{}.map'.format(basename, fmts, shapes, isize))

namepatt = re.compile(r'.*[_\.]([0-9x]+)\.mtx$')
binpatt = re.compile(r'.*[_\.]([0-9x]+)(?:\.[^\.]*\.bin)?$')
mappatt = re.compile(r'.*\.([a-zA-Z0-9]+)_([0-9x]+)\.map$')
typepatt = re.compile(r'([a-zA-Z])')
mapglob = '.[a-zA-Z][0-9][a-zA-Z][0-9][a-zA-Z][0-9]*_[0-9]*[0-9].map'

class BaseSparseMap:
    def __init__(self, mat, shape):
        self.shape = tuple(shape)
        self.mat = mat

    def __repr__(self):
        return '<{} with destination shape ({}), matrix\n  {}>'.format(
            self.__class__.__name__, ', '.join([str(d) for d in self.shape]),
            repr(self.mat))

    def __call__(self, x, axis=None):
        '''Apply SparseMap to as many slots as necessary, starting from axis.
        Resulting new axis will be on the left and reshaped to self.shape.
        '''
        x = np.asanyarray(x)
        if x.size == self.mat.shape[1]:
            return (self.mat*x.flat).reshape(self.shape)
        elif x.ndim == 2 and x.shape[0] == self.mat.shape[1] and axis in [0, None]:
            return (self.mat*x).reshape(self.shape + (x.shape[1],))
        elif axis is None:
            shape = list(x.shape)
            n = 1
            while n < self.mat.shape[1] and shape:
                n *= shape.pop()
            tp = np.promote_types(self.mat.dtype, x.dtype)
            res = np.empty(tuple(shape) + self.mat.shape[:1], tp)
            for idx in np.ndindex(*shape):
                res[idx] = self.mat*x[idx].ravel()
            return res.reshape(tuple(shape) + self.shape)
        else:
            if axis < 0: axis += x.ndim
            shape = list(x.shape)
            n = 1
            while n < self.mat.shape[1]:
                n *= shape.pop(axis)
            n = np.prod(shape)
            n1 = axis
            n2 = x.ndim - len(shape) + n1
            ii = range(x.ndim)
            x1 = x.transpose(ii[n1:n2] + ii[:n1] + ii[n2:]).reshape(self.mat.shape[1], n)
            res = self.mat*x1
            return res.reshape(self.shape + tuple(shape))

    def tomap(self, basename, isize=None, formats=['i4','i4','f8']):
        savemap(basename, self.mat, self.shape, isize, formats)


class CooSparseMap(BaseSparseMap):
    def __init__(self, mat, shape, rshape=None):
        self.shape = tuple(shape)
        self.rshape = rshape
        self.mat = mat

    @classmethod
    def frommm(cls, fname):
        m = namepatt.match(fname)
        shape = map(int, m.group(1).split('x'))[::-1]
        mat = mmread(fname)
        return cls(mat, shape)

    @classmethod
    def fromscrip(cls, fname, shape):
        from scipy.io.netcdf import netcdf_file
        nc = netcdf_file(fname)
        nrow = nc.dimensions['dst_grid_size']
        ncol = nc.dimensions['src_grid_size']
        w = nc.variables['remap_matrix'][:,0]
        i = nc.variables['dst_address'][:] - 1
        j = nc.variables['src_address'][:] - 1
        mat = coo_matrix((w, (i, j)), (nrow, ncol))
        return cls(mat, shape)

    @classmethod
    def fromxesmf(cls, fname, shape, isize):
        from scipy.io.netcdf import netcdf_file
        nc = netcdf_file(fname)
        w = nc.variables['S'][:]
        i = nc.variables['row'][:] - 1
        j = nc.variables['col'][:] - 1
        n = np.prod(shape)
        mat = coo_matrix((w, (i, j)), (n, isize))
        return cls(mat, shape)

    @classmethod
    def frommap(cls, fname, rshape=None):
        m = mappatt.match(fname)
        if m is None:
            l = glob.glob(fname + mapglob)
            if len(l) > 1:
                warn('More than 1 match:\n' + '\n'.join(l))
            elif len(l) == 0:
                raise IOError('SparseMap: ' + fname)
            fname = l[0]
            m = mappatt.match(fname)
        tp,dims = m.groups()
        l = typepatt.split(tp)[1:]
        formats = ['{}{}{}'.format(c in uppercase and '>' or '', c, n)
                   for i in range(0, len(l), 2) for c,n in [l[i:i+2]]]
        tp = dict(names='ijwpabcdefgh'[:len(formats)], formats=formats)
        shape = list(map(int, dims.split('x')))[::-1]
        nrow = reduce(operator.mul, shape[1:])
        ncol = shape[0]
        ijw = np.fromfile(fname, tp).view(np.recarray)
        mat = coo_matrix((ijw['w'], (ijw['i'], ijw['j'])), (nrow, ncol))
        obj = cls(mat, shape[1:], rshape)
        obj.ra = ijw
        return obj

    @classmethod
    def fromijv(cls, i, j, v, shape, isize=None):
        if isize is not None:
            sh = (np.prod(shape), isize)
            mat = coo_matrix((v, (i, j)), shape=sh)
        else:
            mat = coo_matrix((v, (i, j)))
        return cls(mat, shape)

    @property
    def l(self):
        return LeftIndexer(self)

    @property
    def r(self):
        return RightIndexer(self)

    def lflatnonzero(self):
        idx = sorted(set(self.mat.row))
        imap = np.zeros(self.mat.shape[0]) - 1
        imap[idx] = np.arange(len(idx))
        row = imap[self.mat.row]
        mat = coo_matrix((self.mat.data, (row, self.mat.col)))
        return idx, CooSparseMap(mat, (len(idx),))

    def tocsr(self):
        return CsrSparseMap(self.mat.tocsr(), self.shape)

    def lblock(self, ns):
        i = np.unravel_index(self.mat.row, self.shape)
        oshape = []
        o = []
        for d in range(len(i)):
            o.append(i[d]//ns[d])
            oshape.append((self.shape[d] - 1)//ns[d] + 1)

        oshape = tuple(oshape)
        row = np.ravel_multi_index(o, oshape)
        mat = coo_matrix((self.mat.data, (row, self.mat.col)))
        mat.sum_duplicates()
        return CooSparseMap(mat, oshape)


class LeftIndexer(object):
    def __init__(self, coomap):
        self.mat = coomap.mat
        self.shape = coomap.shape

    def __getitem__(self, idx):
        i = np.unravel_index(self.mat.row, self.shape)
        l = map(indices, idx, self.shape)
        oshape = tuple((np.clip(x[1]-x[0], 0, None) - 1) // x[2] + 1 for x in l)

        msk = (i[0] >= l[0][0]) & (i[0] < l[0][1])
        if l[0][2] != 1:
            msk &= (i[0] - l[0][0]) % l[0][2] == 0

        for d in range(1, len(i)):
            msk &= (i[d] >= l[d][0]) & (i[d] < l[d][1])
            if l[d][2] != 1:
                msk &= (i[d] - l[d][0]) % l[d][2] == 0

        mat = coo_matrix((self.mat.data[msk], (self.mat.row[msk], self.mat.col[msk])))
        return CooSparseMap(mat, oshape)


class RightIndexer(object):
    def __init__(self, coomap, rshape=None):
        mat = coomap.mat
        if not hasattr(mat, 'col'):
            mat = mat.tocoo()
        if rshape is None:
            rshape = coomap.rshape
        self.mat = mat
        self.shape = rshape
        self.lshape = coomap.shape

    def __getitem__(self, idx):
        if Ellipsis in idx:
            i = idx.index(Ellipsis)
            idx = idx[:i] + (len(self.shape) - len(idx) + 1)*(slice(None),) + idx[i+1:]

        l = list(map(indices, idx, self.shape))
        oshape = tuple((np.clip(x[1]-x[0], 0, None) - 1) // x[2] + 1 for x in l)

        i = np.unravel_index(self.mat.col, self.shape)
        msk = (i[0] >= l[0][0]) & (i[0] < l[0][1])
        if l[0][2] != 1:
            msk &= (i[0] - l[0][0]) % l[0][2] == 0

        for d in range(1, len(i)):
            msk &= (i[d] >= l[d][0]) & (i[d] < l[d][1])
            if l[d][2] != 1:
                msk &= (i[d] - l[d][0]) % l[d][2] == 0

        col = 0
        for d in range(len(i)):
            col = col*oshape[d] + (i[d][msk] - l[d][0])//l[d][2]

        nrow = np.prod(self.lshape)
        ncol = np.prod(oshape)
        mat = coo_matrix((self.mat.data[msk], (self.mat.row[msk], col)), (nrow, ncol))
        return CooSparseMap(mat, self.lshape, oshape)


class CsrSparseMap(BaseSparseMap):
    def __init__(self, mat, shape):
        self.shape = tuple(shape)
        self.mat = mat

    @classmethod
    def fromijv(cls, i, j, v, shape, isize=None):
        return CooSparseMap.fromijv(i, j, v, shape, isize).tocsr()

    @classmethod
    def frommm(cls, fname):
        return CooSparseMap.frommm(fname).tocsr()

    @classmethod
    def fromscrip(cls, fname, shape):
        coo = CooSparseMap.fromscrip(fname, shape)
        return cls(coo.mat.tocsr(), coo.shape)

    @classmethod
    def fromxesmf(cls, fname, shape, isize):
        coo = CooSparseMap.fromxesmf(fname, shape, isize)
        return cls(coo.mat.tocsr(), coo.shape)

    @classmethod
    def frommap(cls, fname):
        coo = CooSparseMap.frommap(fname)
        return cls(coo.mat.tocsr(), coo.shape)


CooMap = CooSparseMap
CsrMap = CsrSparseMap
SparseMap = CsrSparseMap

fromijv   = CsrSparseMap.fromijv
fromscrip = CsrSparseMap.fromscrip
fromxesmf = CsrSparseMap.fromxesmf
frommap   = CsrSparseMap.frommap
