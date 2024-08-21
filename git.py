def init(repo_name):
    """ Create a repo dir and init .git folder"""
    os.mkdir(repo_name) # create a dir with repo name
    os.mkdir(os.path.join(repo_name, '.git')) #create a dir .git in repo_name
    for name in ['objects', 'refs', 'refs/heads']: # for each name in list
        os.mkdir(os.path.join(repo_name, '.git', name)) # create a dir
    write_file(os.path.join(repo_name, '.git', 'HEAD'), 
               b'ref: refs/heads/master') # checkout master as head
    print('initialized: {} as an empty repository.'.format(repo_name))
