import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import random
import tqdm
import cv2
import numpy as np
from torch import nn
import pytorch_ssim

import utils
from models import Generator, Discriminator
import dataset
from torchsummary import summary
from transforms import (DualCompose,
                        Resize,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        MaskLabel,
                        RandomBrightnessDual,
                        RandomHueDual,
                        )

if __name__ == '__main__':
    device = torch.device("cuda")
    root = Path("../checkpoints")
    epoch_to_use = 44
    use_previous_model = False
    batch_size = 8
    num_workers = 8
    lr = 5.0e-4
    gaussian_std = 0.02
    n_epochs = 1500
    gamma = 0.99

    img_width = 768
    img_height = 768

    display_img_height = 300
    display_img_width = 300

    loss_ratio = 0.5
    train_transform = DualCompose([
        MaskLabel(),
        Resize(w=img_width, h=img_height),
        HorizontalFlip(),
        VerticalFlip(),
        RandomHueDual(limit=0.3),
        RandomBrightnessDual(limit=0.3),
        Normalize()
        ])

    valid_transform = DualCompose([
        MaskLabel(),
        Resize(w=img_width, h=img_height),
        Normalize()
    ])

    input_path = "../datasets/lumi/A/train"
    label_path = "../datasets/lumi/B/train"

    val_input_path = "../datasets/lumi/A/val"
    val_label_path = "../datasets/lumi/B/val"

    input_file_names = utils.read_lumi_filenames(input_path)
    label_file_names = utils.read_lumi_filenames(label_path)

    val_input_file_names = utils.read_lumi_filenames(val_input_path)
    val_label_file_names = utils.read_lumi_filenames(val_label_path)

    train_dataset = dataset.LumiDataset(input_filenames=input_file_names, label_filenames=label_file_names, transform=train_transform)
    val_dataset = dataset.LumiDataset(input_filenames=val_input_file_names, label_filenames=val_label_file_names, transform=valid_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Building Generator
    netG = Generator(num_classes=3, filters_base=16)
    netG = utils.init_net(netG)
    summary(netG, input_size=(3, img_height, img_width))

    # Building Discriminator
    netD = Discriminator(input_nc=3, filter_base=8, num_block=7)
    netD = utils.init_net(netD)
    summary(netD, input_size=(3, img_height, img_width))

    # Optimizer
    G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma, last_epoch=-1)
    D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma, last_epoch=-1)

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

    # Read existing weights for both G and D models
    if use_previous_model:
        G_model_path = model_root / 'G_model_{}.pt'.format(epoch_to_use)
        D_model_path = model_root / 'D_model_{}.pt'.format(epoch_to_use)
        if G_model_path.exists() and D_model_path.exists():
            state = torch.load(str(G_model_path))
            epoch = state['epoch'] + 1
            step = state['step']
            netG.load_state_dict(state['model'])

            state = torch.load(str(D_model_path))
            best_mean_error = state['error']
            model_state = netD.state_dict()
            pre_trained_state = {k: v for k, v in state['model'].items() if k in model_state}
            model_state.update(pre_trained_state)
            netD.load_state_dict(model_state)

            print('Restored model, epoch {}, step {:,}'.format(epoch, step))
        else:
            print('Failed to restore model')
            exit()
    else:
        epoch = 1
        step = 0
        best_mean_error = 0.0

    save = lambda ep, model, model_path, error, optimizer, scheduler: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'error': error,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, str(model_path))

    dataset_length = len(train_loader)
    validate_each = 1
    log = model_root.joinpath('train.log').open('at', encoding='utf8')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mse_gan = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM()

    best_mean_rec_loss = 100
    for epoch in range(epoch, n_epochs + 1):

        mean_D_loss = 0
        mean_G_loss = 0
        mean_recover_loss = 0
        D_losses = []
        G_losses = []
        recover_losses = []

        G_scheduler.step()
        D_scheduler.step()
        for param_group in G_optimizer.param_groups:
            print('Learning rate of G', param_group['lr'])
        for param_group in D_optimizer.param_groups:
            print('Learning rate of D', param_group['lr'])

        netG.train()
        netD.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))

        try:
            for i, (input_images, label_images) in enumerate(train_loader):
                label_images, input_images = label_images.to(device), input_images.to(device)

                # Update Discriminator
                D_optimizer.zero_grad()

                pred_colors = netG(input_images)
                pred_colors = torch.abs(pred_colors)

                C_real = netD(label_images)
                C_fake = netD(pred_colors.detach())

                mean_C_real = torch.mean(C_real, dim=(0,), keepdim=True).expand_as(C_real).detach()
                mean_C_fake = torch.mean(C_fake, dim=(0,), keepdim=True).expand_as(C_fake).detach()
                loss1 = mse_gan(C_real - mean_C_fake, torch.tensor(1.0).cuda().expand_as(C_real))
                loss2 = mse_gan(C_fake - mean_C_real, torch.tensor(-1.0).cuda().expand_as(C_fake))
                loss = 0.5 * (loss1 + loss2)
                loss.backward()
                D_losses.append(loss.item())
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=10)
                D_optimizer.step()

                # Updating Generator
                G_optimizer.zero_grad()

                pred_colors = netG(input_images)
                pred_colors = torch.abs(pred_colors)

                C_real = netD(label_images)
                C_fake = netD(pred_colors)

                mean_C_real = torch.mean(C_real, dim=(0,), keepdim=True).expand_as(C_real).detach()
                mean_C_fake = torch.mean(C_fake, dim=(0,), keepdim=True).expand_as(C_fake).detach()
                loss1 = mse_gan(C_fake - mean_C_real, torch.tensor(1.0).cuda().expand_as(C_fake))
                loss2 = mse_gan(C_real - mean_C_fake, torch.tensor(-1.0).cuda().expand_as(C_real))
                loss3 = utils.calc_loss(pred_colors, label_images, ssim_loss)
                loss = (1.0 - loss_ratio) * 0.5 * (loss1 + loss2) + loss_ratio * loss3
                loss.backward()
                G_losses.append((0.5 * (loss1 + loss2)).item())
                recover_losses.append(loss3.item())
                torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=10)
                G_optimizer.step()

                step += 1
                tq.update(batch_size)
                mean_D_loss = np.mean(D_losses)
                mean_G_loss = np.mean(G_losses)
                mean_recover_loss = np.mean(recover_losses)
                tq.set_postfix(
                    loss=' D={:.5f}, G={:.5f}, Rec={:.5f}'.format(mean_D_loss, mean_G_loss, mean_recover_loss))

                if i == dataset_length - 2:
                    pred_colors = torch.where(pred_colors > 1.0, torch.tensor(1.0).float().cuda(), pred_colors)
                    label_images_cpu = label_images.data.cpu().numpy()
                    pred_color_cpu = pred_colors.data.cpu().numpy()
                    input_images_cpu = input_images.cpu().numpy()

                    color_imgs = []
                    pred_color_imgs = []
                    input_imgs = []
                    for j in range(batch_size):
                        color_img = label_images_cpu[j]
                        pred_color_img = pred_color_cpu[j]
                        input_img = input_images_cpu[j]

                        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
                        pred_color_img = np.moveaxis(pred_color_img, source=[0, 1, 2], destination=[2, 0, 1])
                        input_img = np.moveaxis(input_img, source=[0, 1, 2], destination=[2, 0, 1])

                        color_img = cv2.resize(color_img, dsize=(display_img_height, display_img_width))
                        pred_color_img = cv2.resize(pred_color_img, dsize=(display_img_height, display_img_width))
                        input_img = cv2.resize(input_img, dsize=(display_img_height, display_img_width))
                        color_imgs.append(color_img)
                        pred_color_imgs.append(pred_color_img)
                        input_imgs.append(input_img)

                    final_color = color_imgs[0]
                    final_pred_color = pred_color_imgs[0]
                    final_input_image = input_imgs[0]
                    for j in range(batch_size - 1):
                        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
                        final_pred_color = cv2.hconcat((final_pred_color, pred_color_imgs[j + 1]))
                        final_input_image = cv2.hconcat((final_input_image, input_imgs[j + 1]))

                    final = cv2.vconcat((final_color, final_pred_color, final_input_image))
                    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(str(results_root / 'generated_mask_{epoch}.png'.format(epoch=epoch)),
                                np.uint8(255 * final))

            if epoch % validate_each == 0:
                torch.cuda.empty_cache()
                rec_losses = []
                counter = 0
                with torch.no_grad():
                    for j, (input_images, label_images) in enumerate(val_loader):
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
                            result = cv2.cvtColor(
                                cv2.hconcat((np.uint8(255 * color),
                                             np.uint8(255 * pred_color),
                                             np.uint8(255 * input_img))),
                                cv2.COLOR_BGR2RGB)
                            cv2.imwrite(str(results_root / 'validation_{counter}.png'.format(counter=counter)), result)

                            result = cv2.cvtColor(np.uint8(255 * pred_color), cv2.COLOR_BGR2RGB)
                            counter += 1
                mean_rec_loss = np.mean(rec_losses)
                tq.set_postfix(
                    loss='validation Rec={:.5f}'.format(mean_rec_loss))

                best_mean_rec_loss = mean_rec_loss
                D_model_path = model_root / "D_model_{}.pt".format(epoch)
                G_model_path = model_root / "G_model_{}.pt".format(epoch)
                save(epoch, netD, D_model_path, best_mean_rec_loss, G_optimizer, G_scheduler)
                save(epoch, netG, G_model_path, best_mean_rec_loss, D_optimizer, D_scheduler)

            utils.write_event(log, step, Rec_error=mean_recover_loss)
            utils.write_event(log, step, Dloss=mean_D_loss)
            utils.write_event(log, step, Gloss=mean_G_loss)
            tq.close()
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            tq.close()
            print('Ctrl+C, saving snapshot')
            print('done.')
            exit()
