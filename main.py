import argparse
import datetime
import logging
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

import hln
from datasets.api import Lastfm, Yoochoose, batch_padding

runtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

torch.manual_seed(42)


def evaluate(model, dataloader, klist, rlist):
    recalls, mrrs = defaultdict(int), defaultdict(int)
    gates, gates_count = defaultdict(int), defaultdict(int)

    total_batch_time = 0

    with torch.no_grad():
        for seqs, mask, labs in dataloader:
            batch_size = seqs.size(0)
            lens = mask.sum(1)
            seqs, labs = seqs.cuda(), labs.cuda()

            batch_start_time = time.time()
            labs_p, gate = model(seqs)
            batch_end_time = time.time()
            total_batch_time += batch_end_time - batch_start_time

            for r in rlist:
                gates_ = gate[lens == r]
                if gates_.size(0):
                    gates[r] += gates_.sum(0)[:, :r]
                    gates_count[r] += gates_.size(0)

            ranks = (labs_p > labs_p[range(batch_size),
                     labs].unsqueeze(1)).sum(1) + 1
            for k in klist:
                ranks_in = ranks <= k
                recalls[k] += ranks_in.sum().item()
                mrrs[k] += (1 / ranks[ranks_in].float()).sum().item()

        for r in rlist:
            if gates_count[r]:
                gates[r] = gates[r].cpu() / gates_count[r]
            logger.info("Sessions of length %d in test data: %d",
                        r, gates_count[r])

        for k in klist:
            recalls[k] /= len(dataloader.dataset)
            mrrs[k] /= len(dataloader.dataset)

    logger.info(f"Avg. time per batch: {
                total_batch_time/len(dataloader):.6f} seconds")

    return recalls, mrrs, gates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-dc", type=float, default=0.1)
    parser.add_argument("--lr-dc-step", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--valid-portion", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="yoochoose_r64")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--truncate_steps", type=int, default=19)  # be noted
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--num-prefers", type=int, default=3)
    parser.add_argument("--max-norm", type=float, default=1)
    parser.add_argument("--klist", type=str, default="5,10,15,20")
    parser.add_argument("--rlist", type=str, default="5,10,15")
    args = parser.parse_args()
    logger.info(args)

    klist = [int(k.strip()) for k in args.klist.split(",")]
    rlist = [int(r.strip()) for r in args.rlist.split(",")]

    # When do validation, validset as the testset
    if args.dataset == "yoochoose_r64":
        DATASET = Yoochoose
        train_data_path = "yoochoose/processed/train_r64.pkl"
        test_data_path = "yoochoose/processed/test.pkl"
        total_items = 37484
    elif args.dataset == "yoochoose_r4":
        DATASET = Yoochoose
        train_data_path = "yoochoose/processed/train_r4.pkl"
        test_data_path = "yoochoose/processed/test.pkl"
        total_items = 37484
    elif args.dataset == "lastfm":
        DATASET = Lastfm
        train_data_path = "lastfm/processed/lastfm_train.pkl"
        test_data_path = "lastfm/processed/lastfm_valid.pkl"
        total_items = 39164

    if args.truncate_steps is not None:
        logger.info("Truncate each sequence to recent %d", args.truncate_steps)

    if args.valid:
        data = DATASET(train_data_path, args.truncate_steps)
        length = len(data)
        valid_length = int(round(length * args.valid_portion))
        train_length = length - valid_length
        train_data, test_data = random_split(
            data, [train_length, valid_length])
        logger.info("Using validation dataset as test dataset")
    else:
        train_data = DATASET(train_data_path, args.truncate_steps)
        test_data = DATASET(test_data_path, args.truncate_steps)
    logger.info("Traning data: %d, test data: %d",
                len(train_data), len(test_data))

    # DataLoader
    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch_size,
                             collate_fn=batch_padding,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)
    testloader = DataLoader(test_data,
                            batch_size=args.test_batch_size,
                            collate_fn=batch_padding,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    model = hln.HLN(total_items, args.embed_dim,
                    args.hidden_dim, args.num_prefers)
    model.cuda()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_dc_step, gamma=args.lr_dc)

    batch_idx = 0
    for epoch in range(args.max_epochs):
        model.train()

        epoch_start_time = time.time()

        for i, (seqs, mask, labs) in enumerate(tqdm(trainloader), batch_idx):
            seqs, labs = seqs.cuda(), labs.cuda()
            optimizer.zero_grad()

            labs_p, _ = model(seqs)
            loss = F.nll_loss(labs_p, labs)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

        batch_idx += len(trainloader)

        epoch_end_time = time.time()
        logger.info(f"Total epoch time: {
                    epoch_end_time - epoch_start_time:.4f} seconds")

        torch.save(model.state_dict(), f"hln_checkpoint_epoch{epoch}.pt")

        # Evaluate
        model.eval()
        recalls, mrrs, gates = evaluate(model, testloader, klist, rlist)
        for k in klist:
            logger.info("Epoch %d: recall@%d %f, mrr@%d %f",
                        epoch, k, recalls[k], k, mrrs[k])

        lr_scheduler.step()

    # Print model summary
    summary(model, input_size=(args.train_batch_size,
            args.truncate_steps), dtypes=[torch.long], device='cuda')
