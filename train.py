import argparse
args = argparse.ArgumentParser()
args.add_argument("--config_path", help="Path to Config File", default="configs/train_habitat.yaml")
args = args.parse_args()

from configs import get_cfg_defaults

from trainer import Trainer

if __name__ == "__main__":
    
    cfg = get_cfg_defaults()

    if args.config_path[-4:] == 'yaml':
        cfg.merge_from_file(args.config_path)
    else:
        print("No valid config specified, using default")

    cfg.freeze()
    print(cfg)

    # Train stage 1
    # Assume that it has been trained, and it SAVES all the depth outputs in a 

    train = Trainer(cfg)
    train.train()