from typing import Dict, Type


class Registry:
    def __init__(self):
        self._store: Dict[str, Type] = {}

    def register(self, name: str):

        def deco(cls):
            if name in self._store:
                raise RuntimeError(f"{name} already registered")
            self._store[name] = cls
            return cls

        return deco

    def get(self, name: str):
        if name not in self._store:
            raise RuntimeError(f"{name} not registered")
        return self._store[name]

    def names(self):
        return self._store.keys()



PROCESSOR_REG = Registry()
CHUNKER_REG = Registry()
ENCODER_REG = Registry()

# ------- Evaluation -------
EVALUATOR_REG = Registry()


EMD_BACKBONE_REG = Registry()
GENERATOR_REG = Registry()

