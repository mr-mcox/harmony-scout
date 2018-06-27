import numpy as np


class Creature:
    pass


class PitchClassCreature(Creature):
    @staticmethod
    def conform_genotype(gene):
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

    @staticmethod
    def conform_phenotype(gene, valid_pheno, octave_steps=12):
        gene_mult = gene * octave_steps
        diff = gene_mult - np.expand_dims(valid_pheno, axis=1).repeat(2, axis=1)
        dist = np.sum(np.square(diff), axis=2)
        closest = dist.argmin(axis=0)
        return valid_pheno[closest, :]
