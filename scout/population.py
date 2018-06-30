import numpy as np
from numpy.random import RandomState
import itertools
from random import choices


class Population:
    def __init__(self, creature_factory, evolve_params, random_state=None):
        self.random_state = RandomState() if random_state is None else random_state
        self.creature_factory = creature_factory

        e_params = {"fill": {"target_n": 10}, "cull": {"target_n": 5}}
        if evolve_params is not None:
            e_params.update(evolve_params)
        self.evolve_params = e_params
        self.creatures = list()
        self.generations = 0

    def fill(self):
        for i in range(self.evolve_params["fill"]["target_n"]):
            self.creatures.append(self.creature_factory.from_random())

    def cull(self):
        creatures = self.creatures
        creatures.sort(key=lambda c: c.fitness(), reverse=True)
        target_n = self.evolve_params["cull"]["target_n"]
        self.creatures = creatures[:target_n]

    def evolve(self, to_generation=0):
        evolve_params = self.evolve_params
        gen_types = ["random", "mutate", "crossover"]
        cf = self.creature_factory
        if len(self.creatures) < evolve_params["cull"]["target_n"]:
            self.fill()
            self.cull()
        while self.generations < to_generation:
            n_generate = self.evolve_params["evolve"]["target_n"] - len(self.creatures)
            creatures = self.creatures
            creatures.sort(key=lambda c: c.fitness())
            new_creatures = list()
            for i in range(n_generate):
                gen_type = choices(
                    gen_types, weights=evolve_params["evolve"]["origin_probs"]
                )[0]
                if gen_type == "random":
                    new_creatures.append(cf.from_random())
                elif gen_type == "mutate":
                    new_creatures.append(cf.from_mutation(creatures[0]))
                elif gen_type == "crossover":
                    new_creatures.append(cf.from_crossover(creatures[0:3]))
                else:
                    ValueError(f"{gen_type} is an invalid origin_type")
            self.creatures = creatures + new_creatures
            self.cull()
            self.generations += 1

    def fitness_ptile(self, ptile=0.9):
        scores = [c.fitness() for c in self.creatures]
        return np.percentile(scores, ptile * 100)


class CreatureFactory:
    def __init__(self, creature_class, random_state=None, gene_shape=None, judges=None):
        self.creature_class = creature_class
        self.judges = list() if judges is None else judges
        self.random_state = RandomState() if random_state is None else random_state
        self.gene_shape = (1, 1) if gene_shape is None else gene_shape

    def from_random(self):
        shape = self.gene_shape
        rand = self.random_state
        gene = rand.uniform(size=shape)
        return self.creature_class(gene=gene, judges=self.judges)

    def from_mutation(self, gene):
        rand = self.random_state
        shape = gene.shape
        mutate_at = rand.binomial(1, 0.01, size=shape)
        mutation_amt = rand.normal(scale=0.1, size=shape)
        mutation = mutate_at * mutation_amt
        mutated_gene = gene + mutation
        return self.creature_class(gene=mutated_gene, judges=self.judges)

    def from_crossover(self, genes):
        rand = self.random_state
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
        crossover_gene = np.concatenate(gene_parts)
        return self.creature_class(gene=crossover_gene, judges=self.judges)


class Creature:
    def __init__(self, gene, judges=None):
        self._fitness = None
        self._phenotype = None
        self._genotype = self.conform_genotype(gene)
        self.judges = list() if judges is None else judges

    def fitness(self):
        if self._fitness is None:
            score = 0
            for judge in self.judges:
                score += judge.evaluate(self.phenotype)
            self._fitness = score
        return self._fitness

    def conform_phenotype(self, gene):
        return gene

    def conform_genotype(self, gene):
        return gene

    @property
    def genotype(self):
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
