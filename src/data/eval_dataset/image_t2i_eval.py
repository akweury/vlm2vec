import os
import sys

from datasets import load_dataset
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING
from src.model.processor import VLM_IMAGE_TOKENS
from src.model.processor import process_input_text


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_resolution, model_backbone = kwargs['image_resolution'], kwargs['model_backbone']
    image_root = kwargs['image_root']

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    for qry_inst, qry_text, tgt_inst, tgt_captions, tgt_img_paths in (
            zip(batch_dict['qry_inst'], batch_dict['qry_text'], batch_dict['tgt_inst'], batch_dict['tgt_text'], batch_dict['tgt_img_path'])):
        qry_inst = qry_inst.replace("<|image_1|>", VLM_IMAGE_TOKENS[model_backbone])
        query_text = qry_inst + ' ' + qry_text + '\n'
        if kwargs['dataset_name'] == 'VisDial':
            query_text += '\n' # to be consistent with v1
        query_texts.append([query_text])
        query_images.append([None])

        # subtle target side processing, to stay consistent with v1 eval
        if tgt_captions[0].strip():  # WebQA, EDIS have valid text inputs
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_captions = []
            for tgt_cap in tgt_captions:
                tgt_inst_caption = process_input_text(tgt_inst + ' ' + tgt_cap, model_backbone, text='', add_image_token=True)
                tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'
                tgt_inst_captions.append(tgt_inst_caption)
            cand_texts.append(tgt_inst_captions)
        else:
            tgt_inst = tgt_inst.replace("<|image_1|>", "")
            tgt_inst_caption = process_input_text(tgt_inst, model_backbone, text='', add_image_token=True)
            tgt_inst_caption = tgt_inst_caption.replace(" \n", "\n") + '\n'  # to stay consistent with v1 eval
            cand_texts.append([tgt_inst_caption] * len(tgt_img_paths))
        cand_img_paths = [os.path.join(image_root, tgt_img_path) for tgt_img_path in tgt_img_paths]
        img_list = [{"bytes": [None], "paths": [cand_img_path],
                     "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]} for cand_img_path in cand_img_paths]
        cand_images.append(img_list)
        dataset_infos.append({
            "cand_names": tgt_img_paths,
            "label_name": tgt_img_paths[0],
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "image_t2i"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_image_t2i_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]

    dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution

    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True,
                          batch_size=256, num_proc=4,
                          drop_last_batch=False, load_from_cache_file=False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])

    return dataset, None
