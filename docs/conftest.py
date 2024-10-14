import diffrax
import dynamiqs
import jax
import jax.numpy as jnp
import numpy as np
import optax
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


# sybil configuration
pytest_collect_file = Sybil(
    parsers=[DocTestParser(), PythonCodeBlockParser(), SkipParser()],
    patterns=['*.md'],
    setup=sybil_setup,
).pytest()
