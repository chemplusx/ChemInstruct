from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import click

from . import main

__title__ = 'NERLLaMA'
__version__ = '1.1.0'
__author__ = 'Madhavi Kumari'
__email__ = 'xyzmadhavi@gmail.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024 Madhavi Kumari and contributors'


log = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose debug logging.')
@click.version_option(__version__, '--version', '-V')
@click.help_option('--help', '-h')
@click.pass_context
def cli(ctx, verbose):
    """NERLLaMA command line interface."""
    log.debug('NERLLaMA v%s' % __version__)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    logging.getLogger('requests').setLevel(logging.WARN)
    ctx.obj = {}

cli.add_command(main.nerllama_cli)