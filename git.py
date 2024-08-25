import hashlib
import os
import zlib


def write_file(file_path, content):
    """Write the given content to the file specified by file_path."""
    with open(file_path, 'wb') as file:
        file.write(content)

def init(repo_name):
    """Create a repo dir and init .git folder."""
    os.mkdir(repo_name)  # create a dir with repo name
    os.mkdir(os.path.join(repo_name, '.git'))  # create a dir .git in repo_name
    for name in ['objects', 'refs', 'refs/heads']:  # for each name in list
        os.mkdir(os.path.join(repo_name, '.git', name))  # create a dir
    write_file(os.path.join(repo_name, '.git', 'HEAD'),
               b'ref: refs/heads/master')  # checkout master as head
    print('initialized: {} as an empty repository.'.format(repo_name))

def hash_object(data, obj_type, write=True):
    """
      Compute the SHA-1 hash of the given data as a Git object of the specified type 
      (e.g., blob, commit, or tree). If 'write' is True, the object is compressed 
      and stored in the .git/objects directory following the Git storage structure. 
      Returns the SHA-1 hash as a hexadecimal string.

      Args:
          data (bytes): The content to be stored as a Git object.
          obj_type (str): The type of Git object ('blob', 'commit', 'tree').
          write (bool): If True, the object is written to the .git/objects directory.

      Returns:
          str: The SHA-1 hash of the Git object as a hexadecimal string.
      """
    header = '{} {}'.format(obj_type, len(data)).encode()
    # Combine header, NUL byte, and data into full object content
    full_file_data = header + b'\x00' + data
    # use hexdigest to generate readable hex str rep of sha1 hash
    sha = hashlib.sha1(full_file_data).hexdigest()
    if write:
        path = os.path.join('.git', 'objects', sha[:2], sha[2:])
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_file(path, zlib.compress(full_file_data))
    return sha
