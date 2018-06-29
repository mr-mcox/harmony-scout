import numpy as np
from numpy.random import RandomState
import itertools


class Creature:
    def __init__(self, judges=None, random_state=None):
        self.judges = list() if judges is None else judges
        self.random_state = RandomState() if random_state is None else random_state
        self._proto_gene = None
        self._genotype = None
        self._phenotype = None

    def fitness(self):
        score = 0
        for judge in self.judges:
            score += judge.evaluate(self.phenotype)
        return score

    def conform_phenotype(self, gene):
        return gene

    def conform_genotype(self, gene):
        return gene

    @property
    def genotype(self):
        if self._genotype is None:
            if self._proto_gene is None:
                raise ValueError("No gene creation method has been called")
            self._genotype = self.conform_genotype(self._proto_gene)
        return self._genotype

    @property
    def phenotype(self):
        if self._phenotype is None:
            self._phenotype = self.conform_phenotype(self.genotype)
        return self._phenotype


def conform_normalized_pitch_class(gene):
    gene = np.mod(gene, 1)
    gene.sort(axis=1)
    max_shifts = gene.shape[1]
    n_shifts = 0
    needs_shift = gene.sum(axis=1) >= 1
    while needs_shift.sum() > 0:
        if n_shifts > max_shifts:
            raise AssertionError("Number of logical shifts exceeded")
        shifted = np.roll(gene, shift=1, axis=1)
        shifted[:, 0] = shifted[:, 0] - 1
        stack = np.stack([gene, shifted], axis=2)
        gene = stack[np.arange(gene.shape[0]), :, needs_shift * 1]
        needs_shift = gene.sum(axis=1) >= 1
        n_shifts += 1
    return gene


def pitch_classes_with_pitch(pitches, n=3, octave_steps=12):
    pitch_list = [pitches for i in range(n)]
    combos = np.array([x for x in itertools.product(*pitch_list)])
    assert combos.shape[0] == len(pitches) ** n
    scaled = combos / octave_steps
    conformed = conform_normalized_pitch_class(scaled)
    float_pitch = conformed * octave_steps
    int_pitch = float_pitch.round().astype(int)
    unique = np.unique(int_pitch, axis=0)
    return unique


class PitchClassCreature(Creature):
    def __init__(self, valid_phenotypes=None, **kwargs):
        self.valid_phenotypes = valid_phenotypes
        super().__init__(**kwargs)

    def check_proto_gene_empty(self):
        if self._proto_gene is not None:
            raise ValueError("Proto gene was already set")

    def from_random(self, shape):
        rand = self.random_state
        self.check_proto_gene_empty()
        gene = rand.uniform(size=shape)
        self._proto_gene = gene

    def from_mutation(self, gene):
        rand = self.random_state
        self.check_proto_gene_empty()
        shape = gene.shape
        mutate_at = rand.binomial(1, 0.01, size=shape)
        mutation_amt = rand.normal(scale=0.1, size=shape)
        mutation = mutate_at * mutation_amt
        mutated_gene = gene + mutation
        self._proto_gene = mutated_gene

    def from_crossover(self, genes):
        rand = self.random_state
        self.check_proto_gene_empty()
        n_crossover = rand.binomial(genes[0].shape[0], 0.1)
        crossover_points_float = rand.dirichlet(np.ones(n_crossover))
        crossover_points = crossover_points_float.round().astype(int)
        gene_shift_float = rand.uniform(size=n_crossover) * len(genes)
        gene_shift = np.floor(gene_shift_float).astype(int) + 1
        last_gene = 0
        last_idx = 0
        gene_parts = list()
        for point, shift in zip(crossover_points, gene_shift):
            gene_parts.append(genes[last_gene][last_idx:point])
            last_idx = point
            last_gene += shift
            last_gene = last_gene % len(genes)
        gene_parts.append(genes[last_gene][last_idx:])
        self._proto_gene = np.concatenate(gene_parts)

    def conform_genotype(self, gene):
        return conform_normalized_pitch_class(gene)

    def conform_phenotype(self, gene, octave_steps=12):
        valid_pheno = self.valid_phenotypes
        gene_mult = gene * octave_steps
        diff = gene_mult - np.expand_dims(valid_pheno, axis=1).repeat(
            gene.shape[0], axis=1
        )
        dist = np.sum(np.square(diff), axis=2)
        closest = dist.argmin(axis=0)
        return valid_pheno[closest, :]
