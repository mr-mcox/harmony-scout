from old_scout.spaces import PitchSpace
import numpy as np


class ChordInstance:
    def __init__(self,
                 pitches=None,
                 pitch_space=None,
                 scale_space=None,
                 scale_values=None):
        self._pitches = pitches
        self.pitch_space = PitchSpace() if pitch_space is None else pitch_space
        self.scale_space = scale_space
        self.scale_values = scale_values

    @property
    def pitches(self):
        if self._pitches is not None:
            return self._pitches
        return self.scale_space.pitch_values(self.scale_values)

    @pitches.setter
    def pitches(self, values):
        self._pitches = values


class ClassTemplate:
    def __init__(self, pitch_space=None, scale_space=None):
        self.pitch_space = PitchSpace() if pitch_space is None else pitch_space
        self.scale_space = scale_space

    def generate(self, dependencies=None):
        raise NotImplementedError


class ExactScaleCT(ClassTemplate):
    def __init__(self, scale_values, scale_space):
        super().__init__(self, scale_space)
        self.scale_values = scale_values

    def generate(self, dependencies=None):
        return ChordInstance(
            scale_space=self.scale_space, scale_values=self.scale_values)


class RandomWalkCT(ClassTemplate):
    def __init__(self, scale_space, walk_mean=0, walk_sd=2, stacked_scale=2):
        super().__init__(self, scale_space)
        self.walk_mean = walk_mean
        self.walk_sd = walk_sd
        self.stacked_scale = stacked_scale

    def generate(self, dependencies):
        dep = dependencies[0].scale_values
        walk = np.random.normal(self.walk_mean, self.walk_sd, size=1)
        root = np.round(dep[0] + walk)
        stack = np.random.exponential(self.stacked_scale, size=dep.shape[0]-1) * 2
        scale_values = np.round(np.concatenate([root, stack])).cumsum().astype(int)
        return ChordInstance(
            scale_space=self.scale_space, scale_values=scale_values)

class CopyCT(ClassTemplate):

    def generate(self, dependencies):
        return dependencies[0]