Contribute
==========

Request membership to the project
---------------------------------

If you want to get involved and help us developing the openwind toolbox, you will have to create an account on `gitlab.inria.fr <https://gitlab.inria.fr/>`_

At the moment, following `INRIA's policy <https://gitlab.inria.fr/siteadmin/doc/-/wikis/home#gitlab-accounts>`_\ , you have to request this access through our team. In order to create your account, we need you to send us your Name, Surname and email address to `openwind-contact@inria.fr <mailto:openwind-contact@inria.fr>`_. We promise we won't use it for anything else than granting you an access as developer to our project.


Clone project
-------------

In a terminal, clone the project with the following command:

.. code-block:: shell

  git clone git@gitlab.inria.fr:openwind/openwind.git .

.. hint::

  You may want to add your SSH key to your gitlab account to avoid entering
  your credentials each time you push something to gitlab. To do so, follow
  these instructions :

  `docs.gitlab.com/ee/ssh/#add-an-ssh-key-to-your-gitlab-account <https://docs.gitlab.com/ee/ssh/#add-an-ssh-key-to-your-gitlab-account>`_

  If you didn't use SSH key before, you may have to `check if you have an existing \
  ssh key <https://docs.gitlab.com/ee/ssh/#add-an-ssh-key-to-your-gitlab-account>`_,
  and if not, you will have to `generate one <https://docs.gitlab.com/ee/ssh/#generate-an-ssh-key-pair>`_

.. hint::

  Once you have cloned the repository, you have to go inside it:

  .. code:: shell

    cd openwind/

  To be sure that you are at the right place, you can list files (with the ``ls`` command) and you
  should see

  .. code:: shell

    $ ls
    build                   CHANGELOG.md     dist  examples  meta.yaml  openwind.egg-info  requirements.txt  tbump.toml
    build_requirements.txt  CONTRIBUTORS.md  docs  LICENSE   openwind   README.md          setup.py          tests


Create your branch
------------------

Now that you have cloned our repository, you can create a new branch using git:

.. code-block:: sh

   git checkout -b my-feature-branch

You can name your branch whatever you like, *my_feature_branch* is just an example. **Don't use spaces to name branches**. If you are not familiar with *git*, you can have a look `here <https://www.atlassian.com/git/tutorials/what-is-version-control>`_.

Propose some new features
-------------------------

If you want to propose some new feature, you can ask for `merge request <https://docs.gitlab.com/ee/user/project/merge_requests/>`_.

To do so, push you branch to the openwind repository (once you have been added as a member to the project, see above):

.. code-block:: shell

   git push -u origin my_feature_branch

Then ask for a merge request in `gitlab interface <https://docs.gitlab.com/ee/user/project/merge_requests/>`_\ , it will open a discussion around your new feature, and eventually we will merge it to the master branch.

*Note* : ``git push origin master`` is forbidden (you will not be able to do it)

Create issues
-------------

If you have found some bug, or problems in the software, you can open an `Issue <https://docs.gitlab.com/ee/user/project/issues/>`_
