import sys
from inspect import getmembers, isclass
from importlib import import_module

from .utils import camel2snake


class Repository(object):
    modules = ['competition.featurise',
               'competition.estimators', 
               'competition.pipeline', ]
    object_types = ['calcer',
                    'transformer',
                    'selector',
                    'classifier',
                    'spliter',
                    'search',
                    'pipeline', ]

    def __init__(self) -> None:
        super().__init__()
        self._objects = dict()
        self._register_internal()

    def _register_internal(self):
        for module in map(import_module, self.modules):
            for name, obj in getmembers(module, isclass):
                if (obj.__module__.endswith('base') 
                    or obj.__module__ == __name__):
                    continue
                
                flag = False
                for type in self.object_types:
                    flag |= name.lower().endswith(type)
                if not flag:
                    continue
                
                if hasattr(obj, 'name'):
                    self.register(getattr(obj, 'name'), obj)
                else:
                    self.register(None, obj)

    def register(self, name: str, object: type) -> None:
        if name is None:
            name = camel2snake(object.__name__)
        self._objects[name] = object

    def get_object(self, name: str):
        return self._objects[name]
