import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions

# plot all channel separately
def save_image_after_tokenization(data, H, W, filename, outDir):
    if outDir is None:
        raise Exception('outDir is None')
    
    if filename is None:
        raise Exception('filename is None')
    # filename = filename + '.png'


    # Reshape the data to [H, W, Channel]
    data = data.squeeze()
    channel = data.shape[-1]
    data = data.reshape(H, W, channel)
    # data = data.view(H, W, -1)
    numpy_data = data.detach().cpu().numpy()
    # Calculate the channel-wise mean
    # avg_image = torch.mean(data, dim=2)
    for i in range(numpy_data.shape[2]):
        # avg_image = np.mean(numpy_data, axis=2)

        avg_image = numpy_data[:, :, i] * 255.0
        avg_image = np.array(avg_image, dtype=np.uint8)
        # save_name = f'{filename}+_{i}.jpg'
        img_folder = os.path.join(outDir, filename)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        image_path = os.path.join(img_folder, f'{i}.jpg')
        # avg_image.save(image_path)
        cv2.imwrite(image_path, avg_image)

    # Convert the tensor
    # avg_image = avg_image.cpu()
    # avg_image = transforms.ToPILImage()(avg_image)
    

    

    print(f"Image saved at {image_path} shape:{avg_image.shape}")
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

def visualize_val_sample(epoch, val_loader, model):
    model.eval()
    sum_loss = 0
    m_iou_batches = []
    all_results = []
    unique_values = []
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            imgs = sample['image']      #B, 3, 1024, 2048
            gts = sample['label']       #B, 1024, 2048
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

            imgs_1, imgs_2 = imgs[:, :, :, :1024], imgs[:, :, :, 1024:]
            gts_1, gts_2 = gts[:, :, :1024], gts[:, :, 1024:]
            loss_1, out_1 = model(imgs_1, gts_1)
            loss_2, out_2 = model(imgs_2, gts_2)

            out = torch.cat((out_1, out_2), dim = 3)

            # mean over multi-gpu result
            loss = torch.mean(loss_1) + torch.mean(loss_2)
            #miou using torchmetric library
            m_iou = cal_mean_iou(out, gts)

            score = out[0]      #1, C, H, W --> C, H, W = 19, H, W
            score = torch.exp(score)    
            score = score.permute(1, 2, 0)  #H,W,C
            pred = score.argmax(2)  #H,W
            
            pred = pred.detach().cpu().numpy()
            gts = gts[0].detach().cpu().numpy() #1, H, W --> H, W
            confusionMatrix, labeled, correct = hist_info(config.num_classes, pred, gts)
            results_dict = {'hist': confusionMatrix, 'labeled': labeled, 'correct': correct}
            all_results.append(results_dict)

            m_iou_batches.append(m_iou)

            sum_loss += loss

            # print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
            #         + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
            #         + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.val_print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results)
        # print('all unique class values: ', list(set(unique_values)))

        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")
        print('result: ',result_dict)
        val_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} mean_iou:{result_dict['mean_iou']} ############")
        print(f"########## mean_iou using torchmetric library:{val_mean_iou} ############")
        
        return val_loss, result_dict['mean_iou']