import argparse

from base import build_trainer
from config import CodeTalkerConfig
from datasets import Vocaset, CodeTalkerDataManager
from trainers import CodeTalkerTrainer
from evaluation import CodeTalkerEvaluator


def merge_args(assistant, args):
    if args.gpu:
        assistant.cfg.ENV.GPU = args.gpu

def main(args):
    assistant = CodeTalkerConfig(args.config_file)
    merge_args(assistant, args)
    assistant.cfg.freeze()
    assistant.print_info()

    trainer = build_trainer(assistant)
    # if args.eval_only:
    #     trainer.load_model(args.model_dir, epoch=args.load_epoch)
    #     trainer.test()
    #     return

    if not args.no_train:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', type=str, default='config/codetalker/vocaset/stage1.yaml', help='path to config file'
    )
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu id to use'
    )
    parser.add_argument(
        '--no-train', type=bool, default=False, help='wether to train model'
    )
    args = parser.parse_args()
    main(args)