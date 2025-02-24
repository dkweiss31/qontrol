import functools

import diffrax
import dynamiqs
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import pytest
from sybil import Sybil
from sybil.parsers.doctest import DocTestParser
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

import qontrol


def sybil_setup(namespace):  # noqa ARG001
    namespace['dq'] = dynamiqs
    namespace['dx'] = diffrax
    namespace['jax'] = jax
    namespace['jnp'] = jnp
    namespace['np'] = np
    namespace['optax'] = optax
    namespace['ql'] = qontrol
    namespace['mpl'] = matplotlib.pyplot
    namespace['functools'] = functools


@pytest.fixture(scope='session', autouse=True)
def _mpl_params():
    dynamiqs.plot.utils.mplstyle(dpi=150)
    # use a non-interactive backend for matplotlib, to avoid opening a display window
    matplotlib.use('Agg')


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[DocTestParser(), PythonCodeBlockParser(), SkipParser()],
    patterns=['*.md'],
    setup=sybil_setup,
    fixtures=['_mpl_params'],
).pytest()
