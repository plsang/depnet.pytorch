"""
Collect results from multiple experiments
Requires package pytablewriter to render
Mardown format
"""

from __future__ import print_function
import sys
import os

import json
import argparse
import logging
from datetime import datetime
import itertools
import pytablewriter
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        default='output/model',
        type=str,
        help='path to the model dir')
    parser.add_argument(
        '--split',
        default='dev1',
        type=str,
        help='name of the (test) split')
    parser.add_argument(
        '--metric',
        default='map',
        type=str,
        help='name of the target metric')
    parser.add_argument(
        '--concept_type',
        default='my',
        choices=[
            'my',
            'ex'],
        type=str,
        help='type of concept')

    args = parser.parse_args()
    logger.info('Command-line arguments: %s', args)

    start = datetime.now()
    ###
    cnn_types = ['vgg19', 'resnet152']
    finetunes = ['False', 'True']

    if args.concept_type == 'my':
        concepts = [
            'myconceptsv3',
            'mydepsv4',
            'mydepsprepv4',
            'mypasv4',
            'mypasprepv4']
    elif args.concept_type == 'ex':
        concepts = [
            'exconceptsv3',
            'exdepsv4',
            'exdepsprepv4',
            'expasv4',
            'expasprepv4']
    else:
        raise ValueError('Unknown concept type: %s', args.concept_type)

    runs = list(itertools.product(cnn_types, finetunes))

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = 'results'
    writer.header_list = ['Run'] + concepts
    values = []

    for run in runs:
        exp_name = '_'.join(run)
        row_values = [exp_name]
        for concept in concepts:
            json_file = 'mscoco2014_{}_captions_{}.json'.format(
                args.split, concept)
            json_path = os.path.join(args.model_path, exp_name, json_file)
            with open(json_path) as f:
                scores = json.load(f)
                row_values.append(scores[args.metric])
        values.append(row_values)

    writer.value_matrix = values
    writer.write_table()

    ###
    logger.info('Time: %s', datetime.now() - start)
