# Code is based on facebookresearch/dino
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms

# Import dino package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../lib/dino/"))
import utils as dino_utils

# Import own packagea
from src.data.dataloader import ReturnIndexDataset
from src.models import DINO_backed

TMED_CLASSES = ['A2C', 'A4C', 'PLAX', 'PSAX']


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize([args.img_size, args.img_size], interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # loding data
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    # Define device for copmutations
    device = torch.device("cpu")
    if args.use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")

    # ============ building network ... ============
    model = DINO_backed.load_dino(args.pretrained_weights, device=device,
                                  patch_size=args.patch_size, img_size=args.img_size)
    model.to(device)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if dino_utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = dino_utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = dino_utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T,
                   num_classes=4, num_chunks=100, data_type="TMED"):
    predict_class = {i: 0 for i in range(num_classes)}
    sum_class = {i: 0 for i in range(num_classes)}

    if data_type == "TMED":
        CLASSES = TMED_CLASSES
    else:
        CLASSES = list(range(num_classes))

    top1, top2, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))

        pred_corr = predictions.narrow(1, 0, 1)[correct.narrow(1, 0, 1)]
        for c in predict_class: predict_class[c] += int(sum(pred_corr == c))
        for c in sum_class: sum_class[c] += int(sum(targets == c))

        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top2 = top2 + correct.narrow(1, 0, min(2, k)).sum().item()  # top2 does not make sense if k < 2

        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top2 = top2 * 100.0 / total

    result = {}
    for c in predict_class:
        v = predict_class[c] / sum_class[c] if sum_class[c] != 0 else 0
        print(f"Acc {CLASSES[c]}: {v * 100}")
        result[CLASSES[c]] = v * 100

    return top1, top2, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_classes', default=4, help="Number of classes to train.", type=int)
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
                        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=dino_utils.bool_flag,
                        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
                        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None,
                        help="""If the features have already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--data_path', default='/path/to/dataset/', type=str)
    parser.add_argument('--data', default='TMED', type=str, help='Please specify dataset type tu use.')
    parser.add_argument('--length', default=None, help='Number of frames from video to load.')
    parser.add_argument('--img_size', default=112, type=int, help="The size to resize images to.")
    args = parser.parse_args()

    dino_utils.fix_random_seeds(args.seed)
    dino_utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(dino_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if (args.dump_features is not None) and (not os.path.exists(args.dump_features)):
        os.makedirs(args.dump_features)

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if dino_utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top2, result = knn_classifier(train_features, train_labels,
                                                test_features, test_labels, k,
                                                args.temperature, args.num_classes,
                                                data_type=args.data)
            print(f"{k}-NN classifier result: Top1: {top1}, Top2: {top2}")

    dist.barrier()
