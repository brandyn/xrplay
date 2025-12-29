
import pickle
import os, stat

def file_exists(path):
    "Returns true if path refers to any existing object (including a dead symlink)"
    try:
        st = os.lstat(path)
        return True
    except:
        return False


class InfoFile(object):

    """An InfoFile is just a dictionary stored in a file (via pickle).

    Pretty similar, I suppose, to a shelve, but lightweight and
        made only for small files written out in plain ascii form,
        in whole.

    Access and change this through [] and .get ; don't access
        .items directly.

    No changes are saved until flush() is called.  Revert()
        abandons any changes.

    No locking is performed here, so care should be taken to
        lock at a higher level.

    If the file doesn't exist, it will be created on the first
        flush().

    Don't forget to flush() changes!  No auto flushing is done.

    As a convenience, you can create one of these with filename=None,
        and it will just present an in-memory version of same.
    """
    def __init__(self, filename):
        self.filename = filename
        self.items    = None    # Not loaded yet.

    def __iter__(self):
        self.load()
        return iter(self.items)

    def __contains__(self, key):
        self.load()
        return key in self.items

    def __setitem__(self, key, val):
        self.load()
        self.items[key] = val
        self.dirty = True

    def __getitem__(self, key):
        self.load()
        return self.items[key]

    def __delitem__(self, key):
        self.load()
        del self.items[key]
        self.dirty = True

    def get(self, key, default=None):
        self.load()
        return self.items.get(key, default)

    def revert(self):
        if self.items is not None:
            self.items = None
            del self.dirty  # N/A ; throw exception if we try to use it.

    def load(self):
        "Loads from our file.  Does nothing if already loaded."
        if self.items is None:
            if self.filename and file_exists(self.filename):
                with open(self.filename, 'rb') as fl:
                    items = pickle.load(fl)
                assert isinstance(items, dict)
                self.items = items
            else:
                self.items = {}
            self.dirty = False

    def flush(self, verify=True):
        "Returns True if anything was done, False if there were no changes to flush."

        if self.filename and self.items is not None and self.dirty:

            tempname = temp_filename_for(self.filename)
            with open(tempname, 'wb') as fl:
                pickle.dump(self.items, fl)
            if verify:
                with open(tempname, 'rb') as fl:
                    items = pickle.load(fl)
                assert items == self.items
            os.rename(tempname, self.filename)
            self.dirty = False

            return True
        else:
            return False

