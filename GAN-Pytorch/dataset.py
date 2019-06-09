import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class LumiDataset(Dataset):
    def __init__(self, input_filenames, label_filenames, transform):
        self.input_filenames = input_filenames
        self.label_filenames = label_filenames
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_filename = self.input_filenames[idx]
        label_filename = self.label_filenames[idx]
        input_image = load_image(input_filename)
        label_image = load_image(label_filename)
        augmented_input_image, augmented_label_image = self.transform(image=input_image, mask=label_image)
        return img_to_tensor(augmented_input_image), img_to_tensor(augmented_label_image)
