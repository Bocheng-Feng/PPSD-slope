from __future__ import print_function

__all__ = ['getMainBranch', 'a2z', 'z2a', 'readHlist', 'readHlistToSqlite3', 'SimulationAnalysis', \
        'TargetHalo', 'getDistance', 'iter_grouped_subhalos_indices', 'iterTrees']

import os
import re
import math
import gzip
from io import open
from collections import deque
import numpy as np

import requests
from builtins import range, zip

FLOAT_TYPE = np.float32


def _isstring(s):
    try:
        s + ''
    except TypeError:
        return False
    return True


def a2z(a):
    """
    scale factor to redshift
    """
    return 1.0/a - 1.0


def z2a(z):
    """
    redshift to scale factor
    """
    return 1.0/(1.0+z)


def getMainBranch(iterable, get_num_prog=lambda s: s['num_prog']):
    item = iter(iterable)
    q = deque([(next(item), True)])
    X = []
    while q:
        i, i_mb = q.popleft()
        X.append(i_mb)
        n = get_num_prog(i)
        prog_mb = [i_mb] + [False]*(n-1) if n else []
        q.extend((next(item), mb) for mb in prog_mb)
    return np.array(X, dtype=bool)


class BaseParseFields():
    def __init__(self, header, fields=None):
        if len(header)==0:
            if all(isinstance(f, int) for f in fields):
                self._usecols = fields
                self._formats = [FLOAT_TYPE]*len(fields)
                self._names = ['f%d'%f for f in fields]
            else:
                raise ValueError('header is empty, so fields must be a list '\
                        'of int.')
        else:
            header_s = [self._name_strip(__) for __ in header]
            if not fields:
                self._names = header
                names_s = header_s
                self._usecols = list(range(len(names_s)))
            else:
                if _isstring(fields):
                    fields = [fields]
                self._names = [header[f] if isinstance(f, int) else str(f) \
                        for f in fields]
                names_s = [self._name_strip(__) for __ in self._names]
                wrong_fields = [str(f) for s, f in zip(names_s, fields) \
                        if s not in header_s]
                if wrong_fields:
                    raise ValueError('The following field(s) are not available'\
                            ': %s.\nAvailable fields: %s.'%(\
                            ', '.join(wrong_fields), ', '.join(header)))
                self._usecols = [header_s.index(__) for __ in names_s]
            self._formats = [self._get_format(__) for __ in names_s]

    def parse_line(self, l):
        items = l.split()
        try:
            return tuple(c(items[i]) for i, c in zip(self._usecols, self._formats))
        except Exception as _error:
            print('Something wrong when parsing this line:\n{0}'.format(l))
            raise _error

    def pack(self, X):
        return np.array(X, np.dtype({'names':self._names, \
                'formats':self._formats}))

    def _name_strip(self, s):
        return self._re_name_strip.sub('', s).lower()

    def _get_format(self, s):
        return FLOAT_TYPE if self._re_formats.search(s) is None else np.int64

    _re_name_strip = re.compile(r'\W|_')
    _re_formats = re.compile('^phantom$|^mmp$|id$|^num|num$')


class BaseDirectory:
    def __init__(self, dir_path='.'):
        self.dir_path = os.path.expanduser(dir_path)

        #get file_index
        files = os.listdir(self.dir_path)
        matches = [m for m in [self._re_filename.match(__) for __ in files] if m]
        if not matches:
            raise ValueError('cannot find matching files in this directory: %s.'%(self.dir_path))
        indices = np.array([self._get_file_index(__) for __ in matches])
        s = indices.argsort()
        self.files = [matches[i].group() for i in s]
        self.file_indices = indices[s]

        #get header and header_info
        header_info_list = []
        with open('%s/%s'%(self.dir_path, self.files[0]), 'r') as f:
            for l in f:
                if l[0] == '#':
                    header_info_list.append(l)
                else:
                    break
        if header_info_list:
            self.header_info = ''.join(header_info_list)
            self.header = [self._re_header_remove.sub('', s) for s in \
                    header_info_list[0][1:].split()]
        else:
            self.header_info = ''
            self.header = []

        self._ParseFields = self._Class_ParseFields(self.header, \
                self._default_fields)

    def _load(self, index, exact_index=False, additional_fields=[]):
        p = self._get_ParseFields(additional_fields)
        fn = '%s/%s'%(self.dir_path, self.get_filename(index, exact_index))
        with open(fn, 'r') as f:
            l = '#'
            while l[0] == '#':
                try:
                    l = next(f)
                except StopIteration:
                    return p.pack([])
            X = [p.parse_line(l)]
            for l in f:
                X.append(p.parse_line(l))
        return p.pack(X)

    def _get_file_index(self, match):
        return match.group()

    def get_filename(self, index, exact_index=False):
        if exact_index:
            i = self.file_indices.searchsorted(index)
            if self.file_indices[i] != index:
                raise ValueError('Cannot find the exact index %s.'%(str(index)))
        else:
            i = np.argmin(np.fabs(self.file_indices - index))
        return self.files[i]

    def _get_ParseFields(self, additional_fields):
        if not additional_fields:
            return self._ParseFields
        else:
            return self._Class_ParseFields(self.header, \
                    self._default_fields + list(additional_fields))

    def set_default_fields(self, fields):
        self._default_fields = list(fields)
        self._ParseFields = self._Class_ParseFields(self.header, \
                self._default_fields)

    _re_filename = re.compile('.+')
    _re_header_remove = re.compile('')
    _Class_ParseFields = BaseParseFields
    _default_fields = []
    load = _load


class HlistsDir(BaseDirectory):
    _re_filename = re.compile(r'^hlist_(\d+\.\d+).list$')
    _re_header_remove = re.compile(r'\(\d+\)$')
    _default_fields = ['id', 'upid', 'mvir', 'rvir', 'rs', 'x', 'y', 'z', \
            'vmax', 'vpeak']

    def _get_file_index(self, match):
        return math.log10(float(match.groups()[0]))

    def load(self, z=0, exact_index=False, additional_fields=[]):
        return self._load(math.log10(z2a(z)), exact_index, additional_fields)


class RockstarDir(BaseDirectory):
    _re_filename = re.compile(r'^out_(\d+).list$')
    _re_header_remove = re.compile('')
    _default_fields = ['id', 'mvir', 'rvir', 'rs', 'x', 'y', 'z', 'vmax']

    def _get_file_index(self, match):
        fn = '%s/%s'%(self.dir_path, match.group())
        with open(fn, 'r') as f:
            for l in f:
                if l.startswith('#a '):
                    break
            else:
                raise ValueError('Cannot find the scale factor in this file %s'\
                        %(fn))
        return math.log10(float(l.split()[-1]))

    def load(self, z=0, exact_index=False, additional_fields=[]):
        return self._load(math.log10(z2a(z)), exact_index, additional_fields)


class TreesDir(BaseDirectory):
    _re_filename = re.compile(r'^tree_\d+_\d+_\d+.dat$')
    _re_header_remove = re.compile(r'\(\d+\)$')
    _default_fields = ['scale', 'id', 'num_prog', 'upid', 'mvir', 'rvir', \
            'rs', 'x', 'y', 'z', 'vmax']

    def load(self, tree_root_id, additional_fields=[]):
        p = self._get_ParseFields(additional_fields)
        tree_root_id_str = str(tree_root_id)
        location_file = self.dir_path + '/locations.dat'
        if os.path.isfile(location_file):
            with open(location_file, 'r') as f:
                f.readline()
                for l in f:
                    items = l.split()
                    if items[0] == tree_root_id_str:
                        break
                else:
                    raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
            tree_file = '%s/%s'%(self.dir_path, items[-1])
            with open(tree_file, 'r') as f:
                f.seek(int(items[2]))
                X = []
                for l in f:
                    if l[0] == '#': break
                    X.append(p.parse_line(l))
        else:
            for fn in self.files:
                tree_file = '%s/%s'%(self.dir_path, fn)
                with open(tree_file, 'r') as f:
                    l = '#'
                    while l[0] == '#':
                        try:
                            l = next(f)
                        except StopIteration:
                            raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
                    num_trees = int(l)
                    for l in f:
                        if l[0] == '#' and l.split()[-1] == tree_root_id_str:
                            break #found tree_root_id
                    else:
                        continue #not in this file, check the next one
                    X = []
                    for l in f:
                        if l[0] == '#': break
                        X.append(p.parse_line(l))
                    break #because tree_root_id has found
            else:
                raise ValueError("Cannot find this tree_root_id: %d."%(\
                                    tree_root_id))
        return p.pack(X)


def _generic_open(f, buffering=100000000):
    """
    Returns
    -------
    fp : file handle
    need_to_close : bool
    """
    if hasattr(f, 'read'):
        return f, False
    else:
        if re.match(r'(s?ftp|https?)://', f, re.I):
            r = requests.get(f, stream=True)
            r.raw.decode_content = True
            fp = r.raw
        elif f.endswith('.gz'):
            fp = gzip.open(f, 'r')
        else:
            fp = open(f, 'r', int(buffering))
        return fp, True


def readHlist(hlist, fields=None, buffering=100000000):
    """
    Read the given fields of a hlist file (also works for out_*.list) as a numpy record array.

    Parameters
    ----------
    hlist : str or file obj
        The path to the file (can be an URL) or a file object.
    fields : str, int, array_like, optional
        The desired fields. It can be a list of string or int. If fields is None (default), return all the fields listed in the header.
    buffering : int

    Returns
    -------
    arr : ndarray
        A numpy record array contains the data of the desired fields.

    Example
    -------
    >>> h = readHlist('hlist_1.00000.list', ['id', 'mvir', 'upid'])
    >>> h.dtype
    dtype([('id', '<i8'), ('mvir', '<f8'), ('upid', '<i8')])
    >>> mass_of_hosts = h['mvir'][(h['upid'] == -1)]
    >>> largest_halo_id = h['id'][h['mvir'].argmax()]
    >>> mass_of_subs_of_largest_halo = h['mvir'][(h['upid'] == largest_halo_id)]

    """
    f, need_to_close = _generic_open(hlist, buffering)
    try:
        l = next(f)
        l = l.strip().lstrip('#').lstrip()
        header = [re.sub(r'\(\d+\)$', '', s) for s in l.split()]
        p = BaseParseFields(header, fields)
        X = [p.parse_line(l) for l in f if l[0] != '#']
    finally:
        if need_to_close:
            f.close()
    return p.pack(X)


def readHlistToSqlite3(db, table_name, hlist, fields=None, unique_id=True, buffering=100000000):
    """
    Read the given fields of a hlist file (also works for out_*.list) and save it to a sqlite3 database.

    Parameters
    ----------
    db : sqlite3.Cursor
        A sqlite3.Cursor object.
    hlist : str
        The path to the file.
    fields : str, int, array_like, optional
        The desired fields. It can be a list of string or int. If fields is None (default), return all the fields listed in the header.
    unique_id : bool
    buffering : int

    Returns
    -------
    db : sqlite3.Cursor
        The same cursor object that was given as input.
    """
    f, need_to_close = _generic_open(hlist, buffering)
    try:
        l = next(f)
        l = l.strip().lstrip('#').lstrip()
        header = [re.sub(r'\(\d+\)$', '', s) for s in l.split()]
        p = BaseParseFields(header, fields)
        db_cols = [re.sub(r'\W+', '_', re.sub(r'^\W+|\W+$', '', s)) for s in p._names]
        db_create_stmt = 'create table if not exists %s (%s)'%(table_name, \
                ','.join('%s %s%s'%(name, 'int' if (fmt is int) else 'real', \
                ' unique' if (name == 'id' and unique_id) else '') \
                for name, fmt in zip(db_cols, p._formats)))
        db_insert_stmt = 'insert or replace into %s values (%s)'%(table_name, \
                ','.join(['?']*len(p._names)))
        db.execute(db_create_stmt)
        for l in f:
            if l[0] != '#':
                db.execute(db_insert_stmt, p.parse_line(l))
    finally:
        if need_to_close:
            f.close()
    db.commit()
    return db


def iterTrees(tree_dat, fields=None, buffering=100000000, return_tree_root_id=False):
    """
    Iterate over all the trees in a tree_X_X_X.dat file.

    Parameters
    ----------
    tree_dat : str or file obj
        The path to the file (can be an URL) or a file object.
    fields : str, int, array_like, optional
        The desired fields. It can be a list of string or int. If fields is None (default), return all the fields listed in the header.
    buffering : int

    Returns
    -------
    trees : generator
        A python generator which iterates over all the trees in that file.

    Example
    -------
    >>> trees = iterTrees('tree_0_0_0.dat', ['id', 'mvir', 'upid', 'num_prog'])
    >>> tree = next(trees)
    >>> tree.dtype
    dtype([('id', '<i8'), ('mvir', '<f8'), ('upid', '<i8')])
    >>> mb = tree[getMainBranch(tree)]

    """
    f, need_to_close = _generic_open(tree_dat, buffering)
    try:
        l = next(f)
        if l[0] != '#':
            raise ValueError('File does not have a proper header line.')
        header = [re.sub(r'\(\d+\)$', '', s) for s in l[1:].split()]
        p = BaseParseFields(header, fields)
        while l[0] == '#':
            try:
                l = next(f)
            except StopIteration:
                num_trees = 0
                break
        else:
            try:
                num_trees = int(l)
            except ValueError:
                raise ValueError('File does not have a proper tree_X_X_X.dat format.')

        try:
            l = next(f)
        except StopIteration:
            if num_trees:
                raise ValueError('File does not have a proper tree_X_X_X.dat format.')

        eof = False
        for _ in range(num_trees):
            if eof or l[0] != '#':
                raise ValueError('File does not have a proper tree_X_X_X.dat format.')
            try:
                tree_root_id = int(l.rpartition(' ')[-1])
            except ValueError:
                raise ValueError('File does not have a proper tree_X_X_X.dat format.')
            X = []
            for l in f:
                if l[0] == '#':
                    if return_tree_root_id:
                        yield tree_root_id, p.pack(X)
                    else:
                        yield p.pack(X)
                    break
                X.append(p.parse_line(l))
            else:
                eof = True

    finally:
        if need_to_close:
            f.close()


class SimulationAnalysis:

    def __init__(self, hlists_dir=None, trees_dir=None, rockstar_dir=None):
        self._directories = {}

        if hlists_dir is not None:
            self.set_hlists_dir(hlists_dir)

        if trees_dir is not None:
            self.set_trees_dir(trees_dir)

        if rockstar_dir is not None:
            self.set_rockstar_dir(rockstar_dir)

        if len(self._directories) == 0:
            raise ValueError('Please specify at least one directory.')

    def set_trees_dir(self, trees_dir):
        self._directories['trees'] = TreesDir(trees_dir)
        self._trees = {}
        self._main_branches = {}

    def set_rockstar_dir(self, rockstar_dir):
        self._directories['olists'] = RockstarDir(rockstar_dir)
        self._olists = {}

    def set_hlists_dir(self, hlists_dir):
        self._directories['hlists'] = HlistsDir(hlists_dir)
        self._hlists = {}

    def load_tree(self, tree_root_id=-1, npy_file=None, additional_fields=[]):
        if 'trees' not in self._directories:
            raise ValueError('You must set trees_dir before using this function.')
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file)
            if tree_root_id < 0:
                tree_root_id = data['id'][0]
            elif tree_root_id != data['id'][0]:
                raise ValueError('tree_root_id does not match.')
            self._trees[tree_root_id] = data
        elif tree_root_id not in self._trees:
            self._trees[tree_root_id] = \
                    self._directories['trees'].load(tree_root_id, \
                    additional_fields=additional_fields)
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, self._trees[tree_root_id])
        return self._trees[tree_root_id]

    def load_main_branch(self, tree_root_id=-1, npy_file=None, keep_tree=False, \
            additional_fields=[]):
        if 'trees' not in self._directories:
            raise ValueError('You must set trees_dir before using this function.')
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file)
            if tree_root_id < 0:
                tree_root_id = data['id'][0]
            elif tree_root_id != data['id'][0]:
                raise ValueError('tree_root_id does not match.')
            self._main_branches[tree_root_id] = data
        elif tree_root_id not in self._main_branches:
            t = self._directories['trees'].load(tree_root_id, \
                    additional_fields=additional_fields)
            mb = getMainBranch(t)
            if keep_tree:
                self._trees[tree_root_id] = t
            self._main_branches[tree_root_id] = t[mb]
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, self._main_branches[tree_root_id])
        return self._main_branches[tree_root_id]

    def _choose_hlists_or_olists(self, use_rockstar=False):
        if 'hlists' not in self._directories and \
                'olists' not in self._directories:
            raise ValueError('You must set hlists_dir and/or rockstar_dir'\
                    'before using this function.')
        elif 'olists' not in self._directories:
            if use_rockstar:
                print("Warning: ignore use_rockstar")
            return self._directories['hlists'], self._hlists
        elif use_rockstar or 'hlists' not in self._directories:
            return self._directories['olists'], self._olists
        else:
            return self._directories['hlists'], self._hlists

    def load_halos(self, z=0, npy_file=None, use_rockstar=False, \
            additional_fields=[]):
        d, s = self._choose_hlists_or_olists(use_rockstar)
        fn = d.get_filename(math.log10(z2a(z)))
        if npy_file is not None and os.path.isfile(npy_file):
            data = np.load(npy_file)
            s[fn] = data
        elif fn not in s:
            s[fn] = d.load(z, additional_fields=additional_fields)
        if npy_file is not None and not os.path.isfile(npy_file):
            np.save(npy_file, s[fn])
        return s[fn]

    def del_tree(self, tree_root_id):
        if tree_root_id in self._trees:
            del self._trees[tree_root_id]

    def del_main_branch(self, tree_root_id):
        if tree_root_id in self._main_branches:
            del self._main_branches[tree_root_id]

    def del_halos(self, z, use_rockstar=False):
        d, s = self._choose_hlists_or_olists(use_rockstar)
        fn = d.get_filename(math.log10(z2a(z)))
        if fn in s:
            del s[fn]

    def clear_trees(self):
        self._trees = {}

    def clear_main_branches(self):
        self._main_branches = {}

    def clear_halos(self):
        self._olists = {}
        self._hlists = {}

class TargetHalo:
    def __init__(self, target, halos, box_size=-1):
        self.target = target
        try:
            self.target_id = target['id']
        except KeyError:
            pass
        self.halos = halos
        self.dists = np.zeros(len(halos), float)
        self.box_size = box_size
        half_box_size = 0.5*box_size
        for ax in 'xyz':
            d = halos[ax] - target[ax]
            if box_size > 0:
                d[(d >  half_box_size)] -= box_size
                d[(d < -half_box_size)] += box_size
            self.dists += d*d
        self.dists = np.sqrt(self.dists)

def getDistance(target, halos, box_size=-1):
    t = TargetHalo(target, halos, box_size)
    return t.dists


def iter_grouped_subhalos_indices(host_ids, sub_pids):
    s = sub_pids.argsort()
    k = np.where(sub_pids[s[1:]] != sub_pids[s[:-1]])[0]
    k += 1
    k = np.vstack((np.insert(k, 0, 0), np.append(k, len(s)))).T
    d = np.searchsorted(sub_pids[s[k[:,0]]], host_ids)
    for j, host_id in zip(d, host_ids):
        if j < len(s) and sub_pids[s[k[j,0]]] == host_id:
            yield s[slice(*k[j])]
        else:
            yield np.array([], dtype=int)
