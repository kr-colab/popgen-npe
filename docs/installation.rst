Installation
============

First, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/kr-colab/popgen-npe.git
    cd popgen-npe

Then set up the environment using either uv or conda.

For uv users (faster)
---------------------

.. code-block:: bash

    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -e .

For CPU-only (no CUDA), remove the ``[tool.uv]`` extra-index-url from
``pyproject.toml`` before installing.

For conda users
---------------

.. code-block:: bash

    conda env create -f environment.yaml
    conda activate popgen_npe_env

You are now ready to use popgen-npe for your simulation-based inference tasks in population genetics!
