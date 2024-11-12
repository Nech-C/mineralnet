import random

from PIL import Image
from torchvision.transforms import ( RandomResizedCrop, Compose, Normalize,
ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter)



def crop_image_with_box(img, box):
    """
    Crops the image based on the given bounding box coordinates.
    
    Parameters:
    img (PIL.Image): A PIL Image object.
    box (list): Bounding box coordinates [x_min, y_min, x_max, y_max] as proportions.
    
    Returns:
    cropped_image: Cropped image as a PIL Image object.
    """
    width, height = img.size
    box = box[random.randint(0, len(box) - 1)]['box']
    # Convert proportional coordinates to pixel coordinates
    left = int(box[0] * width)
    top = int(box[1] * height)
    right = int(box[2] * width)
    bottom = int(box[3] * height)

    # Crop the image using the calculated coordinates
    cropped_image = img.crop((left, top, right, bottom))

    return cropped_image


def train_preprocess(batch):
    size = (224, 224)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(20),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    RandomResizedCrop(size),
    ToTensor(),
    normalize,
    ])

    batch['pixel_values'] = [train_transforms(image.convert("RGB")) for image in batch['image']]
    batch['pixel_values'] = [train_transforms(crop_image_with_box(image.convert('RGB'), boxes))
                             for image, boxes in
                             zip(batch['image'], batch['mineral_boxes'])]
    return batch

def preprocess(batch):
    size = (224, 224)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transforms = Compose([
        RandomResizedCrop(size),
        ToTensor(),
        normalize,
    ])
    batch['pixel_values'] = [test_transforms(image.convert("RGB")) for image in batch['image']]

    return batch
