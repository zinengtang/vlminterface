import importlib

import embodied


class Minecraft(embodied.Wrapper):

  def __init__(self, task, *args, **kwargs):
    module, cls = {
        'wood': 'minecraft_flat:Wood',
        'climb': 'minecraft_flat:Climb',
        'diamond': 'minecraft_flat:Diamond',
        'blueprint': 'minecraft_flat:Blueprints'
    }[task].split(':')
    module = importlib.import_module(f'.{module}', __package__)
    cls = getattr(module, cls)
    env = cls(*args, **kwargs)
    super().__init__(env)
