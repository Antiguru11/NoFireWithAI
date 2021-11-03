from typing import Union
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

    def register(self, name: str, object_or_name: Union[type, str]) -> None:
        obj = object_or_name
        if isinstance(obj, str):
            parts = obj.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]

            module = import_module(module_name)
            obj = getattr(module, class_name)

        if name is None:
            name = camel2snake(obj.__name__)
    
        self._objects[name] = obj

    def get_object(self, name: str):
        return self._objects[name]
