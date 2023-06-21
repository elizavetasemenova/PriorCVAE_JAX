try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='priorCVAE',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Prior CVAE',
      author='',
      packages=['priorCVAE']
      )
