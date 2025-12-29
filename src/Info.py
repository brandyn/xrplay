
"""This module just provides a trivial global persistent database
which lives in ~/.cache/xrplay (or ~/.xrplay ; or worst case the
state/ dir next to this src/ dir).

Currently this is used to remember the last view-angle offsets
    per video; and is used by some plugins.

You can get and put data objects by file path (only the hash of
    the final filename is used so same-named videos in different
    directories will be treated as the same video as far as saved
    state goes).
"""
import os

from hashlib   import md5
from FileUtils import InfoFile

homedir = os.environ.get('HOME')

if homedir and os.path.isdir(homedir):
    cachedir = f"{homedir}/.cache"
    if os.path.isdir(cachedir):
        statedir = f"{cachedir}/xrplay"
    else:
        statedir = f"{homedir}/.xrplay"
else:
    statedir = os.path.realpath(f"{__file__}/../../state")

if not os.path.exists(statedir):
    print(f"Creating {statedir}")
    os.mkdir(statedir)

if os.path.isdir(statedir):
    db = InfoFile(f"{statedir}/xrplay.info")
    motion_dir = f"{statedir}/motion"
    if not os.path.exists(motion_dir):
        os.mkdir(motion_dir)
else:
    print(f"WARNING: {statedir} missing or not a directory")
    db         = None
    motion_dir = None

def hash(path):
    return md5(os.path.realpath(path).split('/')[-1].encode()).hexdigest()

def get(path):
    if db is None:
        return {}
    return db.get(hash(path), {})

def put(path, info):
    if db is None:
        return
    db[hash(path)] = info
    db.flush()

def motion_file(path):
    assert motion_dir is not None
    return f"{motion_dir}/{hash(path)}"

