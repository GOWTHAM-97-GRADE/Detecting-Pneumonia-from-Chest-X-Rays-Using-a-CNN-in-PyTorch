import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.dataframe = annotation_file
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.dataframe.iloc[index, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
