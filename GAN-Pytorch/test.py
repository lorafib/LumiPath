import torch
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
import pytorch_ssim

import utils
from models import Generator
import dataset
from torchsummary import summary
from transforms import (DualCompose,
                        Resize,
                        Normalize,
                        MaskLabel
                        )


if __name__ == '__main__':
    device = torch.device("cuda")
    root = Path("../checkpoints")
    epoch_to_use = 44
    use_previous_model = True
    batch_size = 8
    num_workers = 8
    n_epochs = 1500
    gamma = 0.99

    img_width = 768
    img_height = 768

    display_img_height = 300
    display_img_width = 300

    test_transform = DualCompose([
        MaskLabel(),
        Resize(w=img_width, h=img_height),
        Normalize()
    ])

    input_path = "../datasets/lumi/A/test"
    label_path = "../datasets/lumi/B/test"

    input_file_names = utils.read_lumi_filenames(input_path)
    label_file_names = utils.read_lumi_filenames(label_path)

    dataset = dataset.LumiDataset(input_filenames=input_file_names, label_filenames=label_file_names, transform=test_transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Building Generator
    netG = Generator(num_classes=3, filters_base=16)
    netG = utils.init_net(netG)
    summary(netG, input_size=(3, img_height, img_width))

    try:
        model_root = root / "models"
        model_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")

    try:
        results_root = root / "results"
        results_root.mkdir(mode=0o777, parents=False)
    except OSError:
        print("path exists")
    print('Restoring mode')

    # Read existing weights for G model
    G_model_path = model_root / 'model_768_3_17.pth' #G_model_{}.pt'.format(epoch_to_use)
    if G_model_path.exists():
        state = torch.load(str(G_model_path))
        epoch = state['epoch'] + 1
        step = state['step']
        netG.load_state_dict(state['model'])

        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        print('Failed to restore model')
        exit()

    dataset_length = len(loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ssim_loss = pytorch_ssim.SSIM()

    mean_recover_loss = 0
    recover_losses = []

    try:
        rec_losses = []
        counter = 0
        with torch.no_grad():
            for j, (input_images, label_images) in enumerate(loader):
                netG.eval()
                label_images, input_images = label_images.to(device), input_images.to(device)
                pred_label_images = netG(input_images)
                pred_label_images = torch.abs(pred_label_images)
                pred_label_images = torch.where(pred_label_images > 1.0, torch.tensor(1.0).float().cuda(), pred_label_images)

                pred_label_images_cpu = pred_label_images.data.cpu().numpy()
                label_images_cpu = label_images.data.cpu().numpy()
                input_images_cpu = input_images.cpu().numpy()
                rec_losses.append(utils.calc_loss(pred_label_images, label_images, ssim_loss).item())

                for idx in range(label_images_cpu.shape[0]):
                    color = label_images_cpu[idx]
                    pred_color = pred_label_images_cpu[idx]
                    input_img = input_images_cpu[idx]
                    color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                    pred_color = np.moveaxis(pred_color, source=[0, 1, 2], destination=[2, 0, 1])
                    input_img = np.moveaxis(input_img, source=[0, 1, 2], destination=[2, 0, 1])
                    color = cv2.cvtColor(np.uint8(255 * color), cv2.COLOR_BGR2RGB)
                    pred_color = cv2.cvtColor(np.uint8(255 * pred_color), cv2.COLOR_BGR2RGB)
                    input_img = cv2.cvtColor(np.uint8(255 * input_img), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(results_root / 'label_{counter}.png'.format(counter=counter)), color)
                    cv2.imwrite(str(results_root / 'test_{counter}.png'.format(counter=counter)), pred_color)
                    cv2.imwrite(str(results_root / 'input_{counter}.png'.format(counter=counter)), input_img)
                    
                    counter += 1
            mean_rec_loss = np.mean(rec_losses)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print('Ctrl+C, saving snapshot')
        print('done.')
        exit()
    
    print("Mean recovery loss is ", mean_rec_loss)
