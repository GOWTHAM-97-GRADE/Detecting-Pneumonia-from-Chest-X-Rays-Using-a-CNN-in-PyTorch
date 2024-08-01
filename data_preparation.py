import os
import pandas as pd

labels = ['PNEUMONIA', 'NORMAL']

def prepare_dataset(base_dir, labels):
    dataset = []
    for label in labels:
        path = os.path.join(base_dir, label)
        class_num = labels.index(label)
        for img_name in os.listdir(path):
            dataset.append({"img_name": os.path.join(label, img_name), "label": class_num})
    return pd.DataFrame(dataset)

train_df = prepare_dataset("./Dataset/chest_xray/train", labels)
test_df = prepare_dataset("./Dataset/chest_xray/test", labels)
val_df = prepare_dataset("./Dataset/chest_xray/val", labels)
