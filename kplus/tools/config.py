import collections
import configparser as ConfigParser
import errno
import functools
import logging
import optparse
import glob
import os
import sys
import tempfile
import warnings
import kplus

from functools import cached_property
from os.path import expandvars, expanduser, abspath, realpath, normcase
from kplus.tools import appdirs

from typing import Optional


class _MyOption(optparse.Option):
        config = None # must be overriden
        TYPES = ["path", "choice"]
        def __init__(self, *opts, **attrs):
            self.my_default = attrs.pop('my_default', None)
            self.cli_loadable = attrs.pop('cli_loadable', True)
            env_name = attrs.pop('env_name', None)
            self.env_name = env_name or ''
            self.file_loadable = attrs.pop('file_loadable', True)
            self.nargs_ = attrs.get('nargs')
            if self.nargs_ == '?':
                const = attrs.pop('const', None)
                attrs['nargs'] = 1
            attrs.setdefault('metavar', attrs.get('type', 'string').upper())
            super().__init__(*opts, **attrs)
            is_new_option = False
            if self.dest and self.dest not in self.config.options_index:
                self.config.options_index[self.dest] = self
                is_new_option = True
            if self.nargs_ == '?':
                self.const = const
                for opt in self._short_opts + self._long_opts:
                    self.config.optional_options[opt] = self
            if env_name is None and is_new_option and self.file_loadable:
                # generate an env_name for file_loadable settings that are in the index
                self.env_name = 'KPLUS_' + self.dest.upper()
            elif env_name and not is_new_option:
                raise ValueError(f"cannot set env_name to an option that is not indexed: {self}")


class configmanager:
    def __init__(self):
        self._default_options = {}
        self._env_options = {}
        self._cli_options = {}
        self.options = collections.ChainMap(
            self._cli_options,
            self._env_options,
            self._default_options,)
        self.options_index = {}
        self.optional_options = {}
        self.parser = self._build_cli()
        self._load_default_options()
        self._parse_config()
        
    def _build_cli(self):
        MyOption = type('MyOption', (_MyOption,), {'config': self})
        version = "%s %s" % (kplus.Release.description, kplus.Release.version)
        parser = optparse.OptionParser(version=version, option_class=MyOption)
        group = optparse.OptionGroup(parser, "Testing Options")
        group.add_option("--test", dest="test", action="store_true", my_default=False, help="Run the script in a test mode, won't actually upload or update status")
        parser.add_option_group(group)
        group = optparse.OptionGroup(parser, "Logging Options")
        group.add_option("--logfile", dest="logfile", type='path', my_default='',
                         help="file where the server log will be stored")
        levels = ['info', 'warn', 'test', 'critical', 'error',
                  'debug', 'notset']
        group.add_option('--log-level', dest='log_level', type='choice',
                         choices=levels, my_default='info',
                         help='specify the level of the logging. Accepted values: %s.' % (levels,))
        parser.add_option_group(group)
        return parser
        
    def _load_default_options(self):
        self._default_options.clear()
        self._default_options.update({
            option_name: option.my_default
            for option_name, option in self.options_index.items()})
        self._default_options['data_dir'] = (
            appdirs.user_data_dir(kplus.Release.product_name, kplus.Release.author)
            if os.path.isdir(os.path.expanduser('~')) else
            appdirs.site_data_dir(kplus.Release.product_name, kplus.Release.author)
            if sys.platform in ['win32', 'darwin'] else
            f'/var/lib/{kplus.Release.product_name}'
         )

    def parse_config(self, args: Optional[list[str]] = None, *, setup_logging: Optional[bool] = None) -> None:
        from kplus import netsvc
        opt = self._parse_config(args)
        if setup_logging is not False:
            netsvc.setup_logger()
        kplus.env.setup_environment()
        return opt

    def _parse_config(self, args=None):
        for arg_no, arg in enumerate(args or ()):
            if option := self.optional_options.get(arg):
                if arg_no == len(args) - 1 or args[arg_no + 1].startswith('-'):
                    args[arg_no] += '=' + self.format(option.dest, option.const)
                    self._log(logging.DEBUG, "changed %s for %s", arg, args[arg_no])
        opt, unknown_args = self.parser.parse_args(args or [])
        if unknown_args:
            self.parser.error(f"unrecognized parameters: {' '.join(unknown_args)}")
        for option_name in list(vars(opt).keys()):
            if not self.options_index[option_name].cli_loadable:
                delattr(opt, option_name)
        self._load_env_options()
        self._load_cli_options(opt)
        return opt

    def _load_env_options(self):
        self._env_options.clear()
        environ = os.environ
        for option_name, option in self.options_index.items():
            env_name = option.env_name
            if env_name and env_name in environ:
                self._env_options[option_name] = self.parse(option_name, environ[env_name])
    
    def _load_cli_options(self, opt):
        self._cli_options.clear()
        keys = [
            option_name for option_name, option
            in self.options_index.items()
            if option.cli_loadable
            if option.action != 'append']
        for arg in keys:
            if getattr(opt, arg, None) is not None:
                self._cli_options[arg] = getattr(opt, arg)

    def parse(self, option_name, value):
        if not isinstance(value, str):
            e = f"can only cast strings: {value!r}"
            raise TypeError(e)
        if value == 'None':
            return None
        option = self.options_index[option_name]
        if option.action in ('store_true', 'store_false'):
            check_func = self._check_bool
        else:
            check_func = self.parser.option_class.TYPE_CHECKER[option.type]
        return check_func(option, option_name, value)

    def get(self, key, default=None):
        return self.options.get(key, default)

    def __setitem__(self, key, value):
        if isinstance(value, str) and key in self.options_index:
            value = self.parse(key, value)
        self.options[key] = value

    def __getitem__(self, key):
        return self.options[key]

    @cached_property
    def root_path(self):
        return self._normalize(os.path.join(os.path.dirname(__file__), '..'))

    @classmethod
    def _normalize(cls, path):
        if not path:
            return ''
        return normcase(realpath(abspath(expanduser(expandvars(path.strip())))))
    
    @property
    def work_dir(self):
        d = os.path.join(self["data_dir"], 'AllinOneKaraoke by iantirta')
        try:
            os.makedirs(d, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            assert os.access(d, os.W_OK), \
                "%s: directory is not writable" % d
        return d


config = configmanager()
