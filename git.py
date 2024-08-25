#!/usr/bin/python3
import argparse
import collections
import difflib
import hashlib
import enum
import operator
import os
import stat
import struct
import sys
import urllib
import zlib
from datetime import time

# Represents an entry in the Git index file (.git/index)
IndexEntry = collections.namedtuple('IndexEntry', [
    'creation_time_sec', 'creation_time_nsec', 'mod_time_sec', 'mod_time_nsec',
    'device', 'inode', 'permissions', 'user_id', 'group_id', 'file_size', 'sha1_hash', 'entry_flags', 'file_path'
])


class GitObjectType(enum.Enum):
    """Enumerates the types of Git objects."""
    COMMIT = 1
    TREE = 2
    BLOB = 3


def read_bytes_from_file(file_path):
    """Read the binary contents of a file at the specified path."""
    with open(file_path, 'rb') as file:
        return file.read()


def write_bytes_to_file(file_path, data):
    """Write binary data to a file at the specified path."""
    with open(file_path, 'wb') as file:
        file.write(data)

def init(repo):
    """Set up the repository directory and initialize the .git structure."""
    os.mkdir(repo)
    os.mkdir(os.path.join(repo, '.git'))
    subdirs = ['objects', 'refs', 'refs/heads']
    for subdir in subdirs:
        os.mkdir(os.path.join(repo, '.git', subdir))
    write_bytes_to_file(os.path.join(repo, '.git', 'HEAD'), b'ref: refs/heads/master')
    print(f'Repository initialized at: {repo}')

def hash_object(data, obj_type, write=True):
    """Generate a SHA-1 hash for the given object data and optionally store it."""
    header_info = f'{obj_type} {len(data)}'.encode()
    complete_data = header_info + b'\x00' + data
    sha1_hash = hashlib.sha1(complete_data).hexdigest()
    if write:
        object_path = os.path.join('.git', 'objects', sha1_hash[:2], sha1_hash[2:])
        if not os.path.exists(object_path):
            os.makedirs(os.path.dirname(object_path), exist_ok=True)
            write_bytes_to_file(object_path, zlib.compress(complete_data))
    return sha1_hash

def find_object(sha1_prefix):
    """Locate the object with the given SHA-1 prefix and return its path."""
    if len(sha1_prefix) < 2:
        raise ValueError('Prefix must be at least 2 characters long.')
    object_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
    suffix = sha1_prefix[2:]
    matched_files = [file for file in os.listdir(object_dir) if file.startswith(suffix)]
    if not matched_files:
        raise ValueError(f'Object with prefix {sha1_prefix!r} not found.')
    if len(matched_files) > 1:
        raise ValueError(f'Multiple objects found with prefix {sha1_prefix!r}.')
    return os.path.join(object_dir, matched_files[0])

def read_object(sha1_prefix):
    """Read and return the object data and type for the specified SHA-1 prefix."""
    object_path = find_object(sha1_prefix)
    compressed_data = read_bytes_from_file(object_path)
    decompressed_data = zlib.decompress(compressed_data)
    null_byte_position = decompressed_data.index(b'\x00')
    header_content = decompressed_data[:null_byte_position]
    object_type, size_str = header_content.decode().split()
    size = int(size_str)
    object_data = decompressed_data[null_byte_position + 1:]
    if size != len(object_data):
        raise ValueError(f'Expected size {size}, but got {len(object_data)} bytes.')
    return object_type, object_data

def cat_file(mode, sha1_prefix):
    """Output details of the object specified by SHA-1 prefix according to the mode."""
    obj_type, data = read_object(sha1_prefix)
    if mode in ['commit', 'tree', 'blob']:
        if obj_type != mode:
            raise ValueError(f'Expected object type {mode}, but got {obj_type}.')
        sys.stdout.buffer.write(data)
    elif mode == 'size':
        print(len(data))
    elif mode == 'type':
        print(obj_type)
    elif mode == 'pretty':
        if obj_type in ['commit', 'blob']:
            sys.stdout.buffer.write(data)
        elif obj_type == 'tree':
            for mode, path, sha1 in read_tree(data=data):
                entry_type = 'tree' if stat.S_ISDIR(mode) else 'blob'
                print(f'{mode:06o} {entry_type} {sha1}\t{path}')
        else:
            raise ValueError(f'Unhandled object type {obj_type!r}.')
    else:
        raise ValueError(f'Unexpected mode {mode!r}.')


def read_index():
    """Load the git index file and return a list of IndexEntry objects."""
    try:
        index_data = read_bytes_from_file(os.path.join('.git', 'index'))
    except FileNotFoundError:
        return []

    checksum = hashlib.sha1(index_data[:-20]).digest()
    assert checksum == index_data[-20:], 'Index checksum is invalid'

    header_signature, header_version, entry_count = struct.unpack('!4sLL', index_data[:12])
    assert header_signature == b'DIRC', f'Unexpected index signature {header_signature}'
    assert header_version == 2, f'Unsupported index version {header_version}'

    index_entries_data = index_data[12:-20]
    entries = []
    position = 0

    while position + 62 < len(index_entries_data):
        end_of_fields = position + 62
        entry_fields = struct.unpack('!LLLLLLLLLL20sH', index_entries_data[position:end_of_fields])
        path_end_index = index_entries_data.index(b'\x00', end_of_fields)
        file_path = index_entries_data[end_of_fields:path_end_index].decode()

        index_entry = IndexEntry(*(entry_fields + (file_path,)))
        entries.append(index_entry)

        entry_length = ((62 + len(file_path) + 8) // 8) * 8
        position += entry_length

    assert len(entries) == entry_count, 'Entry count mismatch'
    return entries


def ls_files(details=False):
    """List files in the index with optional details (mode, SHA-1, stage)."""
    for entry in read_index():
        if details:
            stage_number = (entry.flags >> 12) & 3
            print(f'{entry.mode:06o} {entry.sha1.hex()} {stage_number}\t{entry.path}')
        else:
            print(entry.path)


def get_status():
    """Determine the status of working files, returning changed, new, and deleted paths."""
    current_paths = set()
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d != '.git']
        for file in files:
            full_path = os.path.join(root, file).replace('\\', '/')
            if full_path.startswith('./'):
                full_path = full_path[2:]
            current_paths.add(full_path)

    indexed_entries = {e.path: e for e in read_index()}
    indexed_paths = set(indexed_entries)

    changed_files = {p for p in (current_paths & indexed_paths)
                     if hash_object(read_bytes_from_file(p), 'blob', write=False) != indexed_entries[p].sha1.hex()}
    new_files = current_paths - indexed_paths
    deleted_files = indexed_paths - current_paths

    return sorted(changed_files), sorted(new_files), sorted(deleted_files)


def status():
    """Display the status of working files."""
    changed_files, new_files, deleted_files = get_status()

    if changed_files:
        print('Changed files:')
        for file_path in changed_files:
            print(f'   {file_path}')

    if new_files:
        print('New files:')
        for file_path in new_files:
            print(f'   {file_path}')

    if deleted_files:
        print('Deleted files:')
        for file_path in deleted_files:
            print(f'   {file_path}')


def diff():
    """Show differences between index and working copy for changed files."""
    changed_files, _, _ = get_status()
    indexed_entries = {e.path: e for e in read_index()}

    for index, file_path in enumerate(changed_files):
        sha1_hash = indexed_entries[file_path].sha1.hex()
        obj_type, file_data = read_object(sha1_hash)
        assert obj_type == 'blob', f'Unexpected object type {obj_type}'

        index_file_lines = file_data.decode().splitlines()
        working_file_lines = read_bytes_from_file(file_path).decode().splitlines()
        diff_lines = difflib.unified_diff(
            index_file_lines, working_file_lines,
            f'{file_path} (index)',
            f'{file_path} (working copy)',
            lineterm=''
        )

        for line in diff_lines:
            print(line)

        if index < len(changed_files) - 1:
            print('-' * 70)


def write_index(entries):
    """Save a list of IndexEntry objects to the git index file."""
    packed_entries = []
    for entry in entries:
        entry_header = struct.pack('!LLLLLLLLLL20sH',
                                   entry.ctime_s, entry.ctime_n, entry.mtime_s, entry.mtime_n,
                                   entry.dev, entry.ino, entry.mode, entry.uid, entry.gid,
                                   entry.size, entry.sha1, entry.flags)
        encoded_path = entry.path.encode()
        entry_length = ((62 + len(encoded_path) + 8) // 8) * 8
        packed_entry = entry_header + encoded_path + b'\x00' * (entry_length - 62 - len(encoded_path))
        packed_entries.append(packed_entry)

    index_header = struct.pack('!4sLL', b'DIRC', 2, len(entries))
    complete_data = index_header + b''.join(packed_entries)
    checksum = hashlib.sha1(complete_data).digest()
    write_bytes_to_file(os.path.join('.git', 'index'), complete_data + checksum)


def add(paths):
    """Include specified file paths in the git index."""
    normalized_paths = [p.replace('\\', '/') for p in paths]
    existing_entries = read_index()
    updated_entries = [e for e in existing_entries if e.path not in normalized_paths]

    for path in normalized_paths:
        sha1_hash = hash_object(read_bytes_from_file(path), 'blob')
        file_stats = os.stat(path)
        path_flags = len(path.encode())
        assert path_flags < (1 << 12)

        new_entry = IndexEntry(
            int(file_stats.st_ctime), 0, int(file_stats.st_mtime), 0,
            file_stats.st_dev, file_stats.st_ino, file_stats.st_mode,
            file_stats.st_uid, file_stats.st_gid, file_stats.st_size,
            bytes.fromhex(sha1_hash), path_flags, path
        )
        updated_entries.append(new_entry)

    updated_entries.sort(key=operator.attrgetter('path'))
    write_index(updated_entries)


def write_tree():
    """Generate a tree object from the current index entries."""
    tree_entries = []
    for entry in read_index():
        assert '/' not in entry.path, 'Only supports single-level directory currently'
        mode_path_str = '{:o} {}'.format(entry.mode, entry.path).encode()
        tree_entry = mode_path_str + b'\x00' + entry.sha1
        tree_entries.append(tree_entry)

    return hash_object(b''.join(tree_entries), 'tree')


def get_local_master_hash():
    """Retrieve the current commit hash (SHA-1) of the local master branch."""
    master_ref_path = os.path.join('.git', 'refs', 'heads', 'master')
    try:
        return read_bytes_from_file(master_ref_path).decode().strip()
    except FileNotFoundError:
        return None


def commit(message, author=None):
    """Commit the current index state to the master branch with the provided message.
    Returns the hash of the commit object.
    """
    tree_hash = write_tree()
    parent_hash = get_local_master_hash()

    if author is None:
        author = '{} <{}>'.format(
            os.environ['GIT_AUTHOR_NAME'], os.environ['GIT_AUTHOR_EMAIL']
        )

    current_time = int(time.mktime(time.localtime()))
    timezone_offset = -time.timezone
    author_timestamp = '{} {}{:02}{:02}'.format(
        current_time,
        '+' if timezone_offset > 0 else '-',
        abs(timezone_offset) // 3600,
        (abs(timezone_offset) // 60) % 60
    )

    commit_lines = [
        'tree ' + tree_hash,
        f'parent {parent_hash}' if parent_hash else None,
        f'author {author} {author_timestamp}',
        f'committer {author} {author_timestamp}',
        '',
        message,
        ''
    ]

    commit_data = '\n'.join(filter(None, commit_lines)).encode()
    commit_sha1 = hash_object(commit_data, 'commit')

    master_ref_path = os.path.join('.git', 'refs', 'heads', 'master')
    write_bytes_to_file(master_ref_path, (commit_sha1 + '\n').encode())

    print('Committed to master: {:7}'.format(commit_sha1))
    return commit_sha1


def extract_lines(data):
    """Extract lines from server response data."""
    extracted_lines = []
    index = 0
    for _ in range(1000):
        length_hex = data[index:index + 4]
        length = int(length_hex, 16)
        line_content = data[index + 4:index + length]
        extracted_lines.append(line_content)
        index += length if length != 0 else 4
        if index >= len(data):
            break
    return extracted_lines


def build_lines_data(lines):
    """Construct a byte string from lines for server transmission."""
    byte_segments = []
    for line in lines:
        line_length = '{:04x}'.format(len(line) + 5).encode()
        byte_segments.append(line_length)
        byte_segments.append(line)
        byte_segments.append(b'\n')
    byte_segments.append(b'0000')
    return b''.join(byte_segments)


def http_request(url, username, password, data=None):
    """Perform an authenticated HTTP request to the specified URL. Uses GET by default; POST if data is provided."""
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, url, username, password)
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(auth_handler)
    response = opener.open(url, data=data)
    return response.read()


def get_remote_master_hash(git_url, username, password):
    """Fetch the commit hash of the remote master branch. Returns SHA-1 hash as hex string or None if no commits."""
    url = f'{git_url}/info/refs?service=git-receive-pack'
    response_data = http_request(url, username, password)
    lines = extract_lines(response_data)
    assert lines[0] == b'# service=git-receive-pack\n'
    assert lines[1] == b''
    if lines[2][:40] == b'0' * 40:
        return None
    sha1_hash, reference = lines[2].split(b'\x00')[0].split()
    assert reference == b'refs/heads/master'
    assert len(sha1_hash) == 40
    return sha1_hash.decode()


def read_tree(sha1=None, data=None):
    """Parse a tree object from SHA-1 (hex string) or data. Returns a list of (mode, path, sha1) tuples."""
    if sha1 is not None:
        obj_type, tree_data = read_object(sha1)
        assert obj_type == 'tree'
    elif data is None:
        raise TypeError('Specify "sha1" or "data"')

    index = 0
    tree_entries = []
    for _ in range(1000):
        delimiter = data.find(b'\x00', index)
        if delimiter == -1:
            break
        mode_path_str = data[index:delimiter].decode()
        mode_str, path = mode_path_str.split()
        mode = int(mode_str, 8)
        sha1_digest = data[delimiter + 1:delimiter + 21]
        tree_entries.append((mode, path, sha1_digest.hex()))
        index = delimiter + 21
    return tree_entries


def find_tree_objects(tree_sha1):
    """Recursively gather SHA-1 hashes of all objects within a tree, including the tree itself."""
    object_hashes = {tree_sha1}
    for mode, path, sha1 in read_tree(sha1=tree_sha1):
        if stat.S_ISDIR(mode):
            object_hashes.update(find_tree_objects(sha1))
        else:
            object_hashes.add(sha1)
    return object_hashes


def find_commit_objects(commit_sha1):
    """Recursively find SHA-1 hashes of all objects in the given commit, including its tree, parents, and the commit itself."""
    objects = {commit_sha1}
    obj_type, commit_data = read_object(commit_sha1)
    assert obj_type == 'commit'

    commit_lines = commit_data.decode().splitlines()
    tree_sha1 = next((line[5:45] for line in commit_lines if line.startswith('tree ')), None)
    if tree_sha1:
        objects.update(find_tree_objects(tree_sha1))

    parent_shas = (line[7:47] for line in commit_lines if line.startswith('parent '))
    for parent_sha1 in parent_shas:
        objects.update(find_commit_objects(parent_sha1))

    return objects


def find_missing_objects(local_sha1, remote_sha1):
    """Determine SHA-1 hashes of objects present in the local commit but missing from the remote commit."""
    local_objects = find_commit_objects(local_sha1)
    if remote_sha1 is None:
        return local_objects
    remote_objects = find_commit_objects(remote_sha1)
    return local_objects - remote_objects


def encode_pack_object(obj_sha1):
    """Encode an object for a pack file, returning bytes with a variable-length header and compressed data."""
    obj_type, obj_data = read_object(obj_sha1)
    type_num = GitObjectType[obj_type].value
    data_size = len(obj_data)

    # Create the variable-length header
    header = []
    size = data_size
    while size > 0:
        byte = (size & 0x7f) | 0x80
        size >>= 7
        if size == 0:
            byte &= 0x7f
        header.append(byte)

    # Return header + compressed data
    return bytes(header) + zlib.compress(obj_data)


def create_pack(objects):
    """Create a pack file for the given set of SHA-1 hashes and return the data bytes of the pack file."""
    header = struct.pack('!4sLL', b'PACK', 2, len(objects))
    body = b''.join(encode_pack_object(sha1) for sha1 in sorted(objects))
    contents = header + body
    sha1_checksum = hashlib.sha1(contents).digest()
    return contents + sha1_checksum


def push(git_url, username=None, password=None):
    """Push the master branch to the specified Git repository URL."""
    if username is None:
        username = os.environ.get('GIT_USERNAME')
    if password is None:
        password = os.environ.get('GIT_PASSWORD')

    remote_sha1 = get_remote_master_hash(git_url, username, password)
    local_sha1 = get_local_master_hash()
    missing_objects = find_missing_objects(local_sha1, remote_sha1)

    print(
        f'Updating remote master from {remote_sha1 or "no commits"} to {local_sha1} ({len(missing_objects)} object{"" if len(missing_objects) == 1 else "s"})')

    lines = [f'{remote_sha1 or "0" * 40} {local_sha1} refs/heads/master\x00 report-status'.encode()]
    data = build_lines_data(lines) + create_pack(missing_objects)

    url = f'{git_url}/git-receive-pack'
    response = http_request(url, username, password, data=data)
    response_lines = extract_lines(response)

    if len(response_lines) < 2:
        raise ValueError(f'Expected at least 2 lines, got {len(response_lines)}')
    if response_lines[0] != b'unpack ok\n':
        raise ValueError(f"Expected 'unpack ok', got: {response_lines[0]}")
    if response_lines[1] != b'ok refs/heads/master\n':
        raise ValueError(f"Expected 'ok refs/heads/master', got: {response_lines[1]}")

    return remote_sha1, missing_objects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', metavar='command')
    subparsers.required = True

    subparsers.add_parser('add', help='Add file(s) to index').add_argument('paths', nargs='+', metavar='path',
                                                                           help='Path(s) of files to add')

    cat_file_parser = subparsers.add_parser('cat-file', help='Display contents of object')
    valid_modes = ['commit', 'tree', 'blob', 'size', 'type', 'pretty']
    cat_file_parser.add_argument('mode', choices=valid_modes, help='Object type or display mode')
    cat_file_parser.add_argument('hash_prefix', help='SHA-1 hash (or hash prefix) of object to display')

    subparsers.add_parser('commit', help='Commit current state of index to master branch').add_argument('-a',
                                                                                                        '--author',
                                                                                                        help='Youssef').add_argument(
        '-m', '--message', required=True, help='Text of commit message')

    subparsers.add_parser('diff', help='Show diff of files changed (between index and working copy)')

    hash_object_parser = subparsers.add_parser('hash-object',
                                               help='Hash contents of given path (and optionally write to object store)')
    hash_object_parser.add_argument('path', help='Path of file to hash')
    hash_object_parser.add_argument('-t', choices=['commit', 'tree', 'blob'], default='blob', dest='type',
                                    help='Type of object')
    hash_object_parser.add_argument('-w', action='store_true', dest='write', help='Write object to object store')

    subparsers.add_parser('init', help='Initialize a new repo').add_argument('repo', help='Directory name for new repo')

    ls_files_parser = subparsers.add_parser('ls-files', help='List files in index')
    ls_files_parser.add_argument('-s', '--stage', action='store_true', help='Show object details')

    push_parser = subparsers.add_parser('push', help='Push master branch to given git server URL')
    push_parser.add_argument('git_url', help='URL of git repo')
    push_parser.add_argument('-p', '--password', help='Password for authentication')
    push_parser.add_argument('-u', '--username', help='Username for authentication')

    subparsers.add_parser('status', help='Show status of working copy')

    args = parser.parse_args()
    if args.command == 'add':
        add(args.paths)
    elif args.command == 'cat-file':
        try:
            cat_file(args.mode, args.hash_prefix)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'commit':
        commit(args.message, author=args.author)
    elif args.command == 'diff':
        diff()
    elif args.command == 'hash-object':
        sha1 = hash_object(read_bytes_from_file(args.path), args.type, write=args.write)
        print(sha1)
    elif args.command == 'init':
        init(args.repo)
    elif args.command == 'ls-files':
        ls_files(details=args.stage)
    elif args.command == 'push':
        push(args.git_url, username=args.username, password=args.password)
    elif args.command == 'status':
        status()
    else:
        raise ValueError(f'Unexpected command: {args.command}')