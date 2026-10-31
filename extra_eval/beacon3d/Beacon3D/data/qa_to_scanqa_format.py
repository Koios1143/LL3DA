import argparse
import json
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, help='scannet | 3rscan | multiscan')
    parser.add_argument('--src', type=str, help='metadata path')
    parser.add_argument('--dst', type=str, help='output path')
    args = parser.parse_args()
    domain = args.domain
    metadata_path = args.src
    output_path = args.dst
    with open(metadata_path) as f:
        metadata = json.load(f)

    id_to_class_path = f'{domain}/obj_id_to_class.json'
    if os.path.exists(id_to_class_path):
        with open(id_to_class_path) as f:
            id_to_class = json.load(f)
    else:
        id_to_class = None

    txt_cache = []
    out = []
    for scene_id, scene_data in tqdm(metadata.items()):
        for base_class in scene_data:
            for obj_id in scene_data[base_class]:
                for i, meta in enumerate(scene_data[base_class][obj_id]['qa']):
                    q = meta['question']
                    a = meta['answer']
                    a_tag = meta['tag']
                    extra = meta['extra_knowledge']
                    cache = f"{scene_id}: {q}"
                    if cache not in txt_cache:
                        this_item = {
                            'question_id': f'{scene_id}_{obj_id}_{i}_{a_tag}_{extra}',
                            'scene_id': scene_id,
                            'question': q,
                            'object_ids': [int(obj_id)],
                            'answers': [a],
                        }
                        if id_to_class:
                            this_item['object_names'] = [id_to_class[scene_id][obj_id]]
                        out.append(this_item)
                        txt_cache.append(cache)

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
