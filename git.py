def init(repo_name):
    """ Create a repo dir and init .git folder"""
    os.mkdir(repo_name) # create a dir with repo name
    os.mkdir(os.path.join(repo_name, '.git')) #create a dir .git in repo_name

