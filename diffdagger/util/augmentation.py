import torchvision.transforms as T
import torchvision.transforms.functional as TVF
import random
from einops import rearrange


def permute_images(tensor):
    return rearrange(tensor, "... h w c -> ... c h w")


def unpermute_images(tensor):
    return rearrange(tensor, "... c h w -> ... h w c")


def center_transform(final_crop_size=(224, 224)):
    return T.Compose(
        [
            # Step 2: Apply a small center crop (slightly smaller than original size)
            T.CenterCrop(size=final_crop_size),
            T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ]
    )


def crop_transform(center_crop_size=(240, 240), final_crop_size=(224, 224)):
    return T.Compose(
        [
            T.RandomCrop(size=final_crop_size),
            T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ]
    )


def rotate_crop_transform(center_crop_size=(240, 240), final_crop_size=(224, 224)):
    return T.Compose(
        [
            T.RandomRotation(degrees=[-5, 5]),
            T.CenterCrop(size=center_crop_size),
            T.RandomCrop(size=final_crop_size),
            T.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ]
    )


def rotate_crop_color_transform(
    center_crop_size=(240, 240), final_crop_size=(224, 224)
):
    return T.Compose(
        [
            T.RandomRotation(degrees=[-5, 5]),
            T.CenterCrop(size=center_crop_size),
            T.RandomCrop(size=final_crop_size),
            T.Normalize(mean=0.0, std=255.0),
            T.Lambda(
                lambda x: TVF.adjust_brightness(
                    x, brightness_factor=random.uniform(0.7, 1.3)
                )
            ),  # Adjust brightness
            T.Lambda(
                lambda x: TVF.adjust_contrast(
                    x, contrast_factor=random.uniform(0.7, 1.3)
                )
            ),  # Adjust contrast
            T.Lambda(
                lambda x: TVF.adjust_saturation(
                    x, saturation_factor=random.uniform(0.7, 1.3)
                )
            ),  # Adjust saturation
            T.Lambda(
                lambda x: TVF.adjust_hue(x, hue_factor=random.uniform(-0.02, 0.02))
            ),  # Adjust hue
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
