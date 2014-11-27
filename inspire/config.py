#!/usr/bin/env python

import os
import logging
import subprocess
import sys
import tempfile


def temp_path(p):
    fd, tmp_path = tempfile.mkstemp(p)
    return tmp_path


def sh_command(cmd):
    subprocess.check_output(cmd, shell=True)


def htk_command(cmd):
    ret = subprocess.call(cmd, shell=True)

    if ret:
        logging.error("HTK command {} exited with error code {}".format(cmd, ret))
        sys.exit(1)


def path(*args, **kwargs):
    p = os.path.join(*args)

    accepted_kwargs = ['create']
    for k in kwargs.keys():
        if k not in accepted_kwargs:
            raise TypeError("path() got an unexpected keyword argument '{}'".format(k))

    if not os.path.exists(p) and kwargs.get('create', False):
        os.makedirs(p)

    return p


def project_path(*args, **kwargs):
    return path(*args, **kwargs)


def which(program):
    import os

    def is_exe(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for pathname in os.environ["PATH"].split(os.pathsep):
            pathname = pathname.strip('"')
            exe_file = os.path.join(pathname, program)

            if is_exe(exe_file):
                return exe_file

    return None


def remove_and_warn_ifexists(fnm):
    if os.path.exists(fnm):
        logging.warning("Overwritting {}".format(fnm))
        os.remove(fnm)


