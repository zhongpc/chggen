"""Class for creating Matscholar embedding."""

import json

import numpy as np



class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = allowed_types
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = self._embedding.keys()

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = embedding.keys()
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance