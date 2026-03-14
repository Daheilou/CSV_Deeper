import os
import json
import h5py
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate train/valid JSON for semi-supervised CSV 2026 challenge")

    parser.add_argument("--root", type=str, default="./data", help="Dataset root path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--val_size", type=int, default=25, help="Total number of samples in validation set")
    parser.add_argument("--val_class1", type=int, default=None, help="Number of class 1 samples in validation set (default: balanced)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    dataset_root_path = args.root
    images_dir_path = os.path.join(dataset_root_path, 'train', 'images')
    labels_dir_path = os.path.join(dataset_root_path, 'train', 'labels')

    # collect image/label filenames in train
    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    train_filename_list = [name for name in all_labeled_filenames]

    # build labeled dataset entries and group by class (cls)
    train_labeled_dataset_list = []
    class0_list = []
    class1_list = []
    for label_filenames in train_filename_list:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        label_h5_file_path = os.path.abspath(os.path.join(labels_dir_path, label_filenames.replace('.h5', '_label.h5')))
        entry = {
            'image': image_h5_file_path,
            'label': label_h5_file_path
        }
        # try to read class label from the label h5 file
        try:
            with h5py.File(label_h5_file_path, 'r') as hf:
                cls_raw = hf['cls'][()]
                cls_val = int(cls_raw)  # ✅ Fixed: directly convert to int
        except Exception as e:
            print(f"Warning: failed to read {label_h5_file_path}, using class 0. Error: {e}")
            cls_val = 0

        train_labeled_dataset_list.append(entry)
        if cls_val == 0:
            class0_list.append(entry)
        else:
            class1_list.append(entry)

    # training set with unlabeled
    train_unlabeled_dataset_list = []
    for label_filenames in all_unlabeled_filenames:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        train_unlabeled_dataset_list.append({
            'image': image_h5_file_path,
            'label': None
        })

    # >>>>>>>>>>>>>>>>>> MODIFIED VALIDATION SPLIT <<<<<<<<<<<<<<<<<<<<
    val_size = args.val_size
    if args.val_class1 is None:
        # Default: balanced (original behavior)
        per_class = val_size // 2
        val_class1 = min(per_class, len(class1_list))
        val_class0 = min(val_size - val_class1, len(class0_list))
    else:
        # User-specified class1 count
        val_class1 = min(args.val_class1, len(class1_list))
        val_class0 = min(val_size - val_class1, len(class0_list))
        if val_class0 < 0:
            val_class0 = 0
            val_class1 = min(val_size, len(class1_list))  # fallback

    # Sample validation set
    sampled0 = random.sample(class0_list, val_class0) if val_class0 > 0 else []
    sampled1 = random.sample(class1_list, val_class1) if val_class1 > 0 else []
    valid_dataset_list = sampled0 + sampled1

    # Remove validation samples from training labeled list
    sampled_image_paths = {e['image'] for e in valid_dataset_list}
    train_labeled_dataset_list = [e for e in train_labeled_dataset_list if e['image'] not in sampled_image_paths]
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # save JSON
    with open(os.path.join(dataset_root_path, 'train_labeled.json'), 'w') as f:
        json.dump(train_labeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'train_unlabeled.json'), 'w') as f:
        json.dump(train_unlabeled_dataset_list, f, indent=4)

    with open(os.path.join(dataset_root_path, 'valid.json'), 'w') as f:
        json.dump(valid_dataset_list, f, indent=4)

    # print dataset statistics
    train_labeled_after_count = len(train_labeled_dataset_list)
    train_unlabeled_count = len(train_unlabeled_dataset_list)

    train_class0 = len([e for e in train_labeled_dataset_list if any('0' in str(h5py.File(e['label'], 'r')['cls'][()]) for _ in [0])]) if train_labeled_after_count else 0
    # Simpler: recompute from lists
    train_class0 = len([e for e in train_labeled_dataset_list if int(h5py.File(e['label'], 'r')['cls'][()]) == 0])
    train_class1 = train_labeled_after_count - train_class0

    val_total = len(valid_dataset_list)
    val_class0_actual = len(sampled0)
    val_class1_actual = len(sampled1)

    train_class0_pct = (train_class0 / train_labeled_after_count * 100) if train_labeled_after_count > 0 else 0.0
    train_class1_pct = (train_class1 / train_labeled_after_count * 100) if train_labeled_after_count > 0 else 0.0
    val_class0_pct = (val_class0_actual / val_total * 100) if val_total > 0 else 0.0
    val_class1_pct = (val_class1_actual / val_total * 100) if val_total > 0 else 0.0

    print("")
    print("=== Dataset split summary ===")
    print("Training set:")
    print(f"  Labeled samples: {train_labeled_after_count}")
    print(f"    - class 0 (low risk): {train_class0} ({train_class0_pct:.1f}%)")
    print(f"    - class 1 (high risk): {train_class1} ({train_class1_pct:.1f}%)")
    print(f"  Unlabeled samples: {train_unlabeled_count}")
    print("")
    print("Validation set:")
    print(f"  Total samples: {val_total}")
    print(f"    - class 0 (low risk): {val_class0_actual} ({val_class0_pct:.1f}%)")
    print(f"    - class 1 (high risk): {val_class1_actual} ({val_class1_pct:.1f}%)")
    print("=============================")