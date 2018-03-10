import argparse
import imagehash
import json
import logging
import math
import numpy as np
import pandas as pd
import re
import time

from google.cloud import datastore, storage, vision
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('project', help='The GCP project')
parser.add_argument('dataset', help='The dataset CSV path')
parser.add_argument("--export_json", type=bool, help='Export label result to JSON')
args = parser.parse_args()

SCORE_THRESHOLD = 0.7
IMAGE_FOLDER = 'images'
MAX_BATCH_SIZE = 9
BUCKET_NAME = 'vision-api-example'
ENTITY_TYPE = 'ImageLabel'
NAMESPACE = 'vision-api-example'

gcs_client = storage.Client(project=args.project)
bucket = gcs_client.get_bucket(BUCKET_NAME)
datastore_client = datastore.Client(project=args.project)
vision_client = vision.ImageAnnotatorClient()


def load_dataset(file_path):
    _df = pd.read_csv(file_path)
    _df['file_path'] = IMAGE_FOLDER + '/' + _df.file
    _df['file_id'] = _df.apply(get_file_id, axis=1)
    _df['image_hash'] = _df.apply(get_image_hash, axis=1)
    keys = [datastore_client.key(ENTITY_TYPE, h, namespace=NAMESPACE) for h in _df['image_hash'].values]
    if keys:
        image_labels = datastore_client.get_multi(keys)
        if image_labels:
            existing_hashes = [ent.key.id_or_name for ent in image_labels]
            _df = _df.loc[~_df['image_hash'].isin(existing_hashes)]
    _df.set_index('file_id')
    return _df


def get_file_id(row):
    p = re.compile('.+/([A-Za-z0-9-_]+).[A-Za-z]+')
    m = p.match(row.file_path)
    _file_id = None
    if m:
        _file_id = m.group(1)
    return _file_id


def get_image_hash(row):
    img = Image.open(row.file_path)
    return str(imagehash.average_hash(img))


def filter_labels(labels):
    filtered = {}
    if labels:
        for l in labels:
            if l.score >= SCORE_THRESHOLD:
                filtered[l.description] = float('{0:.2f}'.format(l.score))
    return filtered


def get_label(_df):
    files = _df.file_path.values
    file_index = {}
    requests = []
    idx = 0
    cols = ['file_path', 'image_labels']
    label_df = pd.DataFrame(columns=cols)
    for f in files:
        try:
            with open(f, 'rb') as image_file:
                request_data = {
                    'image': {'content': image_file.read()},
                    'features': [{'type': vision.enums.Feature.Type.LABEL_DETECTION}]
                }
            requests.append(request_data)
            file_index[f] = idx
            idx += 1
        except Exception as e:
            logger.error(e)
    if requests:
        logger.info('Detect labels for image batch %s' % files)
        batch_response = vision_client.batch_annotate_images(requests)
        if batch_response.responses:
            for k, v in file_index.items():
                labels = filter_labels(batch_response.responses[v].label_annotations)
                label_df = label_df.append(pd.Series([k, labels], index=cols), ignore_index=True)
    return label_df


def upload_to_gcs(_file_id, _file_path):
    gcs_filename = 'gs://%s/%s' % (BUCKET_NAME, _file_id)
    try:
        blob = bucket.blob(_file_id)
        blob.upload_from_filename(filename=_file_path)
        return gcs_filename
    except Exception as e:
        logging.error('Error uploading file %s' % gcs_filename)
        raise e


def store_data(row):
    try:
        gcs_file = upload_to_gcs(row.file_id, row.file_path)
        entity_key = datastore_client.key(ENTITY_TYPE, row.image_hash, namespace=NAMESPACE)
        image_label = datastore.Entity(key=entity_key, exclude_from_indexes=['gcs_file', 'labels'])
        image_label.update({
            'gcs_file': gcs_file,
            'labels': row.image_labels
        })
        if args.export_json:
            with open('output/%s.json' % row.file_id, 'w') as fp:
                label_data = {
                    'imageHash': row.image_hash,
                    'labels': row.image_labels
                }
                json.dump(label_data, fp, indent=2, separators=(',', ': '))
        return image_label
    except Exception as e:
        logging.error('Error storing file data: %s' % row.file_id)


def store_image_label(_df):
    entities = [store_data(row) for index, row in _df.iterrows()]
    if entities:
        entity_partitions = math.ceil(len(entities) / 100)
        entity_batches = np.array_split(entities, entity_partitions)
        for b in entity_batches:
            datastore_client.put_multi(b.tolist())
    else:
        logger.info('No entities to store.')


def tag_images(_df):
    label_df = get_label(_df)
    _df = pd.merge(_df, label_df, left_on='file_path', right_on='file_path')
    store_image_label(_df)


if __name__ == "__main__":
    starttime = time.time()
    df = load_dataset(args.dataset)
    if len(df) > 0:
        partitions = math.ceil(len(df) / MAX_BATCH_SIZE)
        df_batches = np.array_split(df, partitions)
        for d in df_batches:
            tag_images(d)
    endtime = time.time()
    elapsed = endtime - starttime
    print('Elapsed %ss' % int(round(elapsed)))
