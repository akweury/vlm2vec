# Created by MacBook Pro at 25.07.25
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import torch
import os
import wandb
import json
from pathlib import Path
from rtpt import RTPT

from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.utils import batch_to_device
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
import config


def load_images(image_dir, max_images=None):
    image_paths = sorted([f for f in Path(image_dir).glob("*.png")])
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return image_paths


def load_videos(video_dir, max_videos=None):
    video_folders = sorted([f for f in Path(video_dir).iterdir() if f.is_dir()])
    if max_videos is not None:
        video_folders = video_folders[:max_videos]
    videos = []
    for folder in video_folders:
        frame_paths = sorted(folder.glob("frame_*.png"))
        videos.append(frame_paths)
    return videos


def init_wandb(batch_size):
    wandb.init(project="Gestalt-C-Baseline", config={"batch_size": batch_size})


# def run_vlm2vec_legacy(data_path, principle, batch_size, device, img_num, epochs):
#     # Setup model and processor
#     model_args = ModelArguments(
#         model_name='VLM2Vec/VLM2Vec-V2.0',
#         pooling='last',
#         normalize=True,
#         model_backbone='qwen2_vl',
#         lora=True
#     )
#     data_args = DataArguments()
#     processor = load_processor(model_args, data_args)
#     model = MMEBModel.load(model_args).to(device, dtype=torch.bfloat16)
#     model.eval()
#
#     # List of video tasks to evaluate
#     video_tasks = ["task1", "task2", "task3"]  # Replace with actual task names
#     metrics = {}
#
#     for task in video_tasks:
#         # Load task-specific data
#         task_data_path = os.path.join(data_path, task)
#         # Implement task-specific data loading here
#
#         # Prepare batch (adapt to your dataset)
#         processor_inputs = {
#             "text": [...],  # List of texts for this batch
#             "images": [...],  # List of PIL Images for this batch
#         }
#         inputs = Qwen2_VL_process_fn(processor_inputs, processor)
#         inputs = batch_to_device(inputs, device)
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             qry_output = model(qry=inputs)["qry_reps"]
#
#         candidate_texts = [...]  # List of candidate texts
#         candidate_inputs = Qwen2_VL_process_fn({"text": candidate_texts, "images": [None] * len(candidate_texts)}, processor)
#         candidate_inputs = batch_to_device(candidate_inputs, device)
#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             tgt_output = model(tgt=candidate_inputs)["tgt_reps"]
#
#         similarity = model.compute_similarity(qry_output, tgt_output)
#
#         # Compute metrics for this task (e.g., accuracy, recall, etc.)
#         task_metrics = {
#             "accuracy": ...,  # Compute accuracy
#             "recall": ...,  # Compute recall
#             "f1": ...,  # Compute F1 score
#         }
#         metrics[task] = task_metrics
#
#     return metrics

def load_vlm2vec_model(args, device):
    model_args = ModelArguments(
        model_name='Qwen/Qwen2-VL-7B-Instruct',
        checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()

    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, args.use_flash_attn)
    model = model.to(device)
    model.eval()

    return model, processor

def infer_logic_rules_from_img_ilp_tasks(model, processor, train_positive, train_negative, device, principle, batch_size=2):
    def get_reps(images, label):
        reps = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            processor_inputs = {
                "text": [f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'] * len(batch_imgs),
                "images": batch_imgs,
            }
            inputs = Qwen2_VL_process_fn(processor_inputs, processor)
            inputs = batch_to_device(inputs, device)
            with torch.no_grad():
                reps.append(model(qry=inputs)["qry_reps"])
        return torch.cat(reps, dim=0)

    qry_output_pos = get_reps(train_positive, "positive")
    qry_output_neg = get_reps(train_negative, "negative")

    reasoning_text = (
        f"Given these images labeled as {', '.join(['positive'] * len(train_positive) + ['negative'] * len(train_negative))}, "
        f"what is the common logic pattern in the positive images for the principle '{principle}'?"
    )
    reasoning_inputs = processor(
        text=reasoning_text,
        images=None,
        return_tensors="pt"
    )
    reasoning_inputs = {key: value.to(device) for key, value in reasoning_inputs.items()}

    with torch.no_grad():
        output = model.encoder.generate(**reasoning_inputs)
    logic_text = processor.decode(output[0], skip_special_tokens=True)

    return logic_text

# def infer_logic_rules_from_img_ilp_tasks(model, processor, train_positive, train_negative, device, principle):
#     # Batch process positive images
#     processor_inputs_pos = {
#         "text": [f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image',
#                  f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'],
#         "images": train_positive,
#     }
#     inputs_pos = Qwen2_VL_process_fn(processor_inputs_pos, processor)
#     inputs_pos = batch_to_device(inputs_pos, device)
#     with torch.no_grad():
#         qry_output_pos = model(qry=inputs_pos)["qry_reps"]
#         print("qry_output_pos", qry_output_pos)
#     # Batch process negative images
#     processor_inputs_neg = {
#         "text": [f"Represent the given image."] * len(train_negative),
#         "images": train_negative,
#     }
#     inputs_neg = Qwen2_VL_process_fn(processor_inputs_neg, processor)
#     inputs_neg = batch_to_device(inputs_neg, device)
#     with torch.no_grad():
#         qry_output_neg = model(qry=inputs_neg)["qry_reps"]
#         print("qry_output_neg", qry_output_neg)
#     # Prepare reasoning prompt
#     reasoning_text = (
#         f"Given these images labeled as {', '.join(['positive'] * len(train_positive) + ['negative'] * len(train_negative))}, "
#         f"what is the common logic pattern in the positive images for the principle '{principle}'?"
#     )
#     reasoning_inputs = processor(
#         text=reasoning_text,
#         images=None,
#         return_tensors="pt"
#     )
#     reasoning_inputs = {key: value.to(device) for key, value in reasoning_inputs.items()}
#
#     with torch.no_grad():
#         output = model.encoder.generate(**reasoning_inputs)
#     logic_text = processor.decode(output[0], skip_special_tokens=True)
#
#     return logic_text


def infer_logic_rules(model, processor, train_positive, train_negative, device, principle):
    # Collect video representations for all training videos
    video_reps = []
    video_labels = []

    # Process positive videos
    for frames in train_positive:
        inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            rep = model(qry=inputs)["qry_reps"]
        video_reps.append(rep)
        video_labels.append("positive")

    # Process negative videos
    for frames in train_negative:
        inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            rep = model(qry=inputs)["qry_reps"]
        video_reps.append(rep)
        video_labels.append("negative")

    # Stack all video representations
    all_video_reps = torch.cat(video_reps, dim=0)

    # Prepare reasoning input: concatenate video labels as text
    reasoning_text = (
        f"Given these videos labeled as {', '.join(video_labels)}, "
        f"what is the common logic pattern in the positive videos for the principle '{principle}'?"
    )

    # Use the model to reason and generate the logic pattern
    reasoning_inputs = processor(
        text=reasoning_text,
        videos=None,
        return_tensors="pt"
    )
    reasoning_inputs = {key: value.to(device) for key, value in reasoning_inputs.items()}
    with torch.no_grad():
        output = model.generate(**reasoning_inputs)
    logic_text = processor.decode(output[0], skip_special_tokens=True)
    #
    # messages = conversations.vlm2vec_messages(train_positive, train_negative, principle)
    #
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.', videos=video_inputs, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    # inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    # qry_output = model(qry=inputs)["qry_reps"]
    #
    # string = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
    # inputs = processor(text=string, images=None, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # tgt_output = model(tgt=inputs)["tgt_reps"]
    # print(string, '=', model.compute_similarity(qry_output, tgt_output))
    #
    # string = 'A person dressed in a blue jacket shovels the snow-covered pavement outside their house.'
    # inputs = processor(text=string, images=None, return_tensors="pt")
    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
    # tgt_output = model(tgt=inputs)["tgt_reps"]
    # print(string, '=', model.compute_similarity(qry_output, tgt_output))

    return logic_text


def evaluate_vlm2vec_image(model, processor, test_images, logic_rules, device, principle, threshold=0.5):
    y_true = []
    y_pred = []
    similarities = []

    # Prepare logic rule representation
    logic_inputs = processor(
        text=logic_rules,
        images=None,
        return_tensors="pt"
    )
    logic_inputs = {key: value.to(device) for key, value in logic_inputs.items()}
    with torch.no_grad():
        logic_rep = model(tgt=logic_inputs)["tgt_reps"]

    for img, label in test_images:
        # Prepare image representation
        image_inputs = processor(
            text=f"{VLM_IMAGE_TOKENS[QWEN2_VL]}  Represent the given image with the following question: What is in the image",
            images=img,
            return_tensors="pt"
        )
        image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
        with torch.no_grad():
            image_rep = model(qry=image_inputs)["qry_reps"]
            print("image_rep", image_rep)

        # Compute similarity
        similarity = model.compute_similarity(image_rep, logic_rep).item()
        similarities.append(similarity)
        pred = 1 if similarity >= threshold else 0
        y_pred.append(pred)
        y_true.append(label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, f1, precision, recall


def evaluate_vlm2vec(model, processor, test_images, logic_rules, device, principle, threshold=0.5):
    y_true = []
    y_pred = []
    similarities = []

    # Prepare logic rule representation
    logic_inputs = processor(
        text=logic_rules,
        images=None,
        return_tensors="pt"
    )
    logic_inputs = {key: value.to(device) for key, value in logic_inputs.items()}
    with torch.no_grad():
        logic_rep = model(tgt=logic_inputs)["tgt_reps"]

    for frames, label in test_images:
        # Prepare video representation
        video_inputs = processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=[frames],
            return_tensors="pt"
        )
        video_inputs = {key: value.to(device) for key, value in video_inputs.items()}
        video_inputs['pixel_values_videos'] = video_inputs['pixel_values_videos'].unsqueeze(0)
        video_inputs['video_grid_thw'] = video_inputs['video_grid_thw'].unsqueeze(0)
        with torch.no_grad():
            video_rep = model(qry=video_inputs)["qry_reps"]

        # Compute similarity
        similarity = model.compute_similarity(video_rep, logic_rep).item()
        similarities.append(similarity)
        pred = 1 if similarity >= threshold else 0
        y_pred.append(pred)
        y_true.append(label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, f1, precision, recall


def run_vlm2vec_video(data_path, principle, batch_size, device, img_num, epochs):
    init_wandb(batch_size)
    model, processor = load_vlm2vec_model(device)
    principle_path = Path(data_path)

    pattern_folders = sorted(
        [f for f in (principle_path / "train").iterdir() if f.is_dir() and not f.name.startswith('.')]
    )

    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    for pattern_folder in pattern_folders:
        train_positive_videos = load_videos(pattern_folder / "positive", img_num)
        train_negative_videos = load_videos(pattern_folder / "negative", img_num)
        test_positive_videos = load_videos((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative_videos = load_videos((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        # Flatten videos to list of frame paths for each video
        train_positive = [frames for frames in train_positive_videos]
        train_negative = [frames for frames in train_negative_videos]
        test_positive = [frames for frames in test_positive_videos]
        test_negative = [frames for frames in test_negative_videos]

        logic_rules = infer_logic_rules(model, processor, train_positive, train_negative, device, principle)

        test_images = [(frames, 1) for frames in test_positive] + [(frames, 0) for frames in test_negative]
        accuracy, f1, precision, recall = evaluate_vlm2vec(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "logic_rules": logic_rules,
            "precision": precision,
            "recall": recall
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / f"deepseek_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Evaluation complete. Results saved to evaluation_results.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1


def run_vlm2vec_image(args, device, data_path):
    principle = args.principle
    batch_size = args.batch_size
    img_num = args.img_num
    epochs = args.epochs

    init_wandb(batch_size)
    model, processor = load_vlm2vec_model(args, device)
    principle_path = Path(data_path)

    pattern_folders = sorted(
        [f for f in (principle_path / "train").iterdir() if f.is_dir() and not f.name.startswith('.')]
    )

    total_accuracy, total_f1 = [], []
    results = {}
    total_precision_scores = []
    total_recall_scores = []

    for pattern_folder in pattern_folders:
        print(f"Processing pattern folder: {pattern_folder.name}")
        train_positive_images = load_images(pattern_folder / "positive", img_num)
        train_negative_images = load_images(pattern_folder / "negative", img_num)
        test_positive_images = load_images((principle_path / "test" / pattern_folder.name) / "positive", img_num)
        test_negative_images = load_images((principle_path / "test" / pattern_folder.name) / "negative", img_num)

        train_positive = [Image.open(img_path) for img_path in train_positive_images]
        train_negative = [Image.open(img_path) for img_path in train_negative_images]
        test_positive = [Image.open(img_path) for img_path in test_positive_images]
        test_negative = [Image.open(img_path) for img_path in test_negative_images]
        logic_rules = infer_logic_rules_from_img_ilp_tasks(model, processor, train_positive, train_negative, device, principle)

        test_images = [(img, 1) for img in test_positive] + [(img, 0) for img in test_negative]
        accuracy, f1, precision, recall = evaluate_vlm2vec_image(model, processor, test_images, logic_rules, device, principle)

        results[pattern_folder.name] = {
            "accuracy": accuracy,
            "f1_score": f1,
            "logic_rules": logic_rules,
            "precision": precision,
            "recall": recall
        }
        total_accuracy.append(accuracy)
        total_f1.append(f1)
        total_precision_scores.append(precision)
        total_recall_scores.append(recall)

    avg_accuracy = sum(total_accuracy) / len(total_accuracy) if total_accuracy else 0
    avg_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0

    results["average"] = {"accuracy": avg_accuracy, "f1_score": avg_f1}
    results_path = Path(data_path) / f"vlm2vec_image_{principle}.json"
    with open(results_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Image evaluation complete. Results saved to vlm2vec_image_{principle}.json.")
    print(f"Overall Average Accuracy: {avg_accuracy:.2f}% | Average F1 Score: {avg_f1:.4f}")
    wandb.finish()
    return avg_accuracy, avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models with CUDA support.")
    parser.add_argument("--principle", type=str, required=True, help="Specify the principle to filter data.")
    parser.add_argument("--device_id", type=int, help="Specify GPU device ID. If not provided, CPU will be used.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Enable FlashAttention if supported.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    if args.device_id is not None and torch.cuda.is_available():
        device = f"cuda:{args.device_id}"
    else:
        device = "cpu"

    data_path = config.raw_patterns / args.principle
    # List of baseline models
    print("data_path", data_path)
    run_vlm2vec_image(args, device, data_path)

    print("All model evaluations completed.")
