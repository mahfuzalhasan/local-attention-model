import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# plot all channel separately
def save_image_after_tokenization(data, H, W, filename, outDir):
    if outDir is None:
        raise Exception('outDir is None')
    
    if filename is None:
        raise Exception('filename is None')
    filename = filename + '.png'


    # Reshape the data to [H, W, Channel]
    data = data.view(H, W, -1)

    # Calculate the channel-wise mean
    avg_image = torch.mean(data, dim=2)

    # Convert the tensor
    avg_image = avg_image.cpu()

    avg_image = transforms.ToPILImage()(avg_image)
    

    image_path = os.path.join(outDir, filename)
    avg_image.save(image_path)

    print(f"Image saved at {image_path}")
    return avg_image


def save_input_image(data, filename, outDir):
    if outDir is None:
        raise Exception('outDir is None')
    
    if filename is None:
        raise Exception('filename is None')
    filename = filename + '.png'

    # Convert the tensor to a PIL image
    input_image = transforms.ToPILImage()(data[0].cpu())

    

    
    # Save the image to the output folder
    image_path = os.path.join(outDir, filename)
    input_image.save(image_path)

    print(f"Input Image saved at {image_path}")
    return input_image


def save_after_block(data, filename, outDir):
    if outDir is None:
        raise Exception('outDir is None')
    
    if filename is None:
        raise Exception('filename is None')
    filename = filename + '.png'
    # Reshape the data to [H, W, Channel]
    data = data.view(data.shape[2], data.shape[3], data.shape[1])

    # take average from all channels
    avg_image = torch.mean(data, dim=2)

    # Convert the tensor to a PIL image
    avg_image = avg_image.cpu()
    
    avg_image = transforms.ToPILImage()(avg_image)

    image_path = os.path.join(outDir, filename)
    avg_image.save(image_path)

    print(f"Output Image saved at {image_path}")
    return avg_image