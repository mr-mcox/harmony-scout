from scout.sequencer import build_modules, Sequencer
import yaml

def main():
    # import config
    with open('config/simple-1.yaml') as fh:
        config = yaml.load(fh)
    seq = Sequencer(**config['sequencer'])
    build_modules(configs=config['modules'], sequencer=seq)
    seq.sequence()
    pitch_class_judges = seq.for_level['pitch_class']
