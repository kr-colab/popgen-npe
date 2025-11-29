Contributing
============

We welcome contributions to popgen-npe! Here's how you can help:

Reporting Issues
----------------

If you encounter a bug or have a feature request, please open an issue on GitHub:
https://github.com/kr-colab/popgen-npe/issues

When reporting bugs, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Your environment (OS, Python version, package versions)
- Any relevant error messages or logs

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/popgen-npe.git
       cd popgen-npe

3. Create the development environment:

   .. code-block:: bash

       conda env create -f environment.yaml
       conda activate popgen-npe_env

4. Create a branch for your changes:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

Running Tests
-------------

Before submitting changes, ensure all tests pass:

.. code-block:: bash

    pytest tests/

For parallel test execution:

.. code-block:: bash

    pytest -n auto tests/

Submitting Changes
------------------

1. Commit your changes with clear, descriptive commit messages
2. Push to your fork
3. Open a pull request against the main repository
4. Ensure CI checks pass

Code Style
----------

- Follow PEP 8 guidelines for Python code
- Add docstrings for new functions and classes
- Include type hints where appropriate 