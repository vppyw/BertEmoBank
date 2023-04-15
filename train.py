import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

import torch

import utils
import models

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--load_model', type=str)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--early_stop', type=int, default=-1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    return parser.parse_args()

def main(args):
    train_loader, valid_loader, test_loader = utils.get_split_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.load_model != None:
        model = torch.load(
            args.load_model,
            map_location='cpu'
        ).to(args.device)
    else:
        model = models.BertVAD(dropout=args.dropout)
    opt = utils.get_optimizer(args, model)

    loss_fn = utils.get_loss_fn(args)

    best_log = None
    cnt = 0
    pbar = tqdm(range(args.epoch), ncols=50)
    for epoch in pbar:
        log = {'epoch': epoch}
        # Train
        train_log = utils.train(args, model, opt, loss_fn, train_loader)
        for l in train_log.keys():
            log['train_' + l] = train_log[l]

        # Validation
        with torch.no_grad():
            valid_log = utils.test(
                            args, model, loss_fn, valid_loader
                        )
        for l in valid_log:
            log['val_'+l] = valid_log[l]
        if best_log == None or \
           best_log['loss'] > valid_log['loss']:
            best_log = valid_log
            torch.save(model, args.model_file)
            cnt = 0
        else:
            cnt += 1
            if cnt > args.early_stop and args.early_stop > 0:
                break

        # Test
        with torch.no_grad():
            test_log = utils.test(args, model, loss_fn, test_loader)
        for l in test_log:
            log['test_'+l] = test_log[l]
        
        # Write the log file
        with open(args.log_file, 'r') as f:
            logs = json.load(f)
        logs.append(log)
        with open(args.log_file, 'w') as f:
            f.write(json.dumps(logs, indent=2))

if __name__ == '__main__':
    args = parse_args()
    utils.prepare_env(args)
    main(args)
