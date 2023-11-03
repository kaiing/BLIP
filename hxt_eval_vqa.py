'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
import torchvision.transforms as standard_transforms




@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    if config['inference']=='rank':
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate')

            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())
                result.append({"question_id":ques_id, "answer":answer})

        elif config['inference']=='rank':
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})

    return result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####

    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)
    print("dddd")

    model.eval()

    result = []

    img_path = "/data/hxt/01.projects/01.pyProjects/01.multimodel/01.imgs/baby.png"
    image = Image.open(img_path).convert('RGB')

    newsize = (480, 480)
    image = image.resize(newsize)

    transform1 = standard_transforms.ToTensor()
    image_tensor = transform1(image)

    image_tensor = image_tensor.unsqueeze(0)
    image = image_tensor.to(device, non_blocking=True)


    batch_questions = ["what is the person doing?", "how many person in the picture?", "the person's age?"]


    batch_images = torch.cat([image, image, image], 0)


    config['inference'] = 'generate'
    if config['inference'] == 'generate':
        answers = model(batch_images, batch_questions, train=False, inference='generate')
        print('answers:', answers)
        # for answer, ques_id in zip(answers, question_id):
        #     ques_id = int(ques_id.item())
        #     result.append({"question_id": ques_id, "answer": answer})
        #




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)