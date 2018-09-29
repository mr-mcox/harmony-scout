import numpy as np
from numpy.random import RandomState
import itertools
from collections import defaultdict
from math import floor


def population_factory(
    population_class, creature_factory, evolve_params=None, random_state=None
):
    def build(parent=None):
        return population_class(
            creature_factory=creature_factory,
            evolve_params=evolve_params,
            random_state=random_state,
            parent=parent,
        )

    return build


class Population:
    def __init__(
        self, creature_factory, evolve_params=None, random_state=None, parent=None
    ):
        self.random_state = RandomState() if random_state is None else random_state
        self.creature_factory = creature_factory
        self.parent = parent

        e_params = {
            "fill": {"target_n": 10},
            "cull": {"target_n": 5},
            "evolve": {"target_n": 20, "origin_probs": [0.5, 0.3, 0.2]},
        }
        if evolve_params is not None:
            e_params.update(evolve_params)
        self.evolve_params = e_params
        self.creatures = list()
        self.generations = 0

    def fill(self):
        for i in range(self.evolve_params["fill"]["target_n"]):
            self.creatures.append(self.creature_factory.from_random(parent=self.parent))

    def cull(self):
        creatures = self.creatures
        creatures.sort(key=lambda c: c.fitness(), reverse=True)
        target_n = self.evolve_params["cull"]["target_n"]
        self.creatures = creatures[:target_n]

    def evolve(self, to_generation=0):
        rand = self.random_state
        evolve_params = self.evolve_params
        gen_types = np.array(["random", "mutate", "crossover"])
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
                gen_type = rand.choice(
                    gen_types, p=evolve_params["evolve"]["origin_probs"]
                )
                if gen_type == "random":
                    new_creatures.append(cf.from_random(parent=self.parent))
                elif gen_type == "mutate":
                    idx = np.floor(rand.uniform(size=1) ** 2 * len(creatures)).astype(
                        int
                    )[0]
                    new_creatures.append(
                        cf.from_mutation(creatures[idx], parent=self.parent)
                    )
                elif gen_type == "crossover":
                    idxs = np.floor(rand.uniform(size=3) ** 2 * len(creatures)).astype(
                        int
                    )
                    sources = list()
                    for i in np.nditer(idxs):
                        sources.append(creatures[i])
                    new_creatures.append(cf.from_crossover(sources, parent=self.parent))
                else:
                    ValueError(f"{gen_type} is an invalid origin_type")
            self.creatures = creatures + new_creatures
            self.cull()
            self.generations += 1
        for creature in self.creatures:
            creature.evolve_sub_population(to_generation=to_generation)

    def fitness_ptile(self, ptile=0.9):
        scores = [c.fitness() for c in self.creatures]
        return np.percentile(scores, ptile * 100)


class CreatureFactory:
    def __init__(
        self,
        creature_class,
        creature_kwargs=None,
        random_state=None,
        gene_shape=None,
        judges=None,
        sub_population_factory=None,
    ):
        self.creature_class = creature_class
        self.creature_kwargs = dict() if creature_kwargs is None else creature_kwargs
        self.judges = list() if judges is None else judges
        self.random_state = RandomState() if random_state is None else random_state
        self.gene_shape = (1, 1) if gene_shape is None else gene_shape
        self.sub_population_factory = sub_population_factory

    def create_creature_from_gene(self, gene, parent):
        return self.creature_class(
            gene=gene,
            judges=self.judges,
            population_factory=self.sub_population_factory,
            parent=parent,
            **self.creature_kwargs
        )

    def from_random(self, parent=None):
        shape = self.gene_shape
        rand = self.random_state
        gene = rand.uniform(size=shape)
        return self.create_creature_from_gene(gene, parent)

    def from_mutation(self, creature, parent=None):
        gene = creature.genotype
        rand = self.random_state
        shape = gene.shape
        mutate_at = rand.binomial(1, 0.01, size=shape)
        mutation_amt = rand.normal(scale=0.1, size=shape)
        mutation = mutate_at * mutation_amt
        mutated_gene = gene + mutation
        return self.create_creature_from_gene(mutated_gene, parent)

    def from_crossover(self, creatures, parent=None):
        genes = [c.genotype for c in creatures]
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
        return self.create_creature_from_gene(crossover_gene, parent)


class Creature:
    def __init__(self, gene, judges=None, population_factory=None, parent=None):
        self._fitness = None
        self._phenotype = None
        self._genotype = self.conform_genotype(gene)
        self.judges = list() if judges is None else judges
        self.population_factory = population_factory
        self.sub_population = None
        self.parent = parent

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

    def evolve_sub_population(self, to_generation=0):
        if self.population_factory is not None:
            self.sub_population = self.population_factory(parent=self)
            self.sub_population.evolve(to_generation=to_generation)


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

def normalize_pitch_class( pitches, octave_steps=12):
    pitch_classes = pitches / octave_steps
    pitch_classes = conform_normalized_pitch_class(pitch_classes) * octave_steps
    pitch_classes = pitch_classes.round().astype(int)
    return pitch_classes


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


class VoicingCreature(Creature):
    def __init__(
        self, valid_phenotypes=None, octave_steps=12, start_voice=None, **kwargs
    ):
        self.valid_phenotypes = valid_phenotypes
        self.octave_steps = octave_steps
        self.voices_for = self.compute_voices_for(valid_phenotypes)
        self.start_voice = start_voice
        super().__init__(**kwargs)

    def compute_voices_for(self, pitches):
        voice_dict = defaultdict(list)
        octave_steps = self.octave_steps
        pitch_classes = normalize_pitch_class( pitches, octave_steps)
        for i in range(pitches.shape[0]):
            pitch_class = tuple(pitch_classes[i].tolist())
            voice_dict[pitch_class].append(pitches[i])
        for key, values in voice_dict.items():
            voice_dict[key] = np.stack(values, axis=0)
        return voice_dict



    def conform_genotype(self, gene):
        return np.mod(gene, 1)

    def conform_phenotype(self, gene):
        previous = self.start_voice
        voices = list()
        for i in range(self.parent.phenotype.shape[0]):
            pc_tuple = tuple(self.parent.phenotype[i].tolist())
            options = self.voices_for[pc_tuple]
            option_num = floor(gene[i] * len(options))
            diff = np.square(options - previous).sum(axis=1)
            idx = np.argsort(diff)[option_num]
            choice = options[idx]
            voices.append(choice)
            previous = choice
        return np.stack(voices, axis=0)
