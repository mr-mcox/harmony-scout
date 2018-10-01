from scout.sequencer import build_modules, Sequencer
from scout.population import pitch_classes_with_pitch, CreatureFactory, Population, PitchClassCreature
import yaml

def main():
    # import config
    with open('config/simple-1.yaml') as fh:
        config = yaml.load(fh)
    seq = Sequencer(**config['sequencer'])
    build_modules(configs=config['modules'], sequencer=seq)
    seq.sequence()
    pitch_class_judges = seq.for_level['pitch_class']
    valid_pitches = pitch_classes_with_pitch([0, 2, 4, 5, 7, 9, 11], n=4)
    cf = CreatureFactory(PitchClassCreature, {'valid_phenotypes': valid_pitches}, gene_shape=(6, 4), judges=pitch_class_judges)
    p = Population(cf)
    p.evolve(to_generation=5)
    print(p.creatures[0].phenotype)

if __name__ == '__main__':
    main()

