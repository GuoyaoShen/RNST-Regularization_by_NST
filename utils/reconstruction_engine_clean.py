import numpy as np

from utils.data import *
from utils.evaluation import *
from utils.optimizer import *
from net.style_transfer import *



def RNST_engine(
        path_content,
        path_style,
        path_ref,
        idx_pulse,
        idx_slice,
        std_noise,
        engine_settings=(20, 0.15, 0.7, 5, 3),  #[N_iter,miu,lam,update_num_steps,num_style_step]
        random_seed=None,
        show_steps=False,
        show_info=False,
        use_hybrid_loss=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ====== construct data ======
    # load img
    if idx_slice+1 < 10:
        path_p_h = path_style + '/IMG000' + str(idx_slice+1) + '.dcm'  # start from 1
        path_ref = path_ref + '/IMG000' + str(idx_slice+1) + '.dcm'  # start from 1
    else:
        path_p_h = path_style + '/IMG00' + str(idx_slice+1) + '.dcm'
        path_ref = path_ref + '/IMG00' + str(idx_slice+1) + '.dcm'
    if idx_pulse != 2:
        path_p_l = path_content + '/vs001_' + str(idx_slice) + '.dcm'  # start from 0
    else:
        path_p_l = path_content + '/vs001_' + str(idx_slice+80) + '.dcm'  # start from 80

    p_h = load_img(path_p_h)  # high field img
    p_l = load_img(path_p_l)  # low field img
    p_ref = load_img(path_ref)  # reference img, registered scan

    # concatenate as RGB images
    p_h = p_h[np.newaxis, ...]
    p_l = p_l[np.newaxis, ...]
    content_img = np.concatenate((p_l, p_l, p_l), axis=0)  # low2high
    style_img = np.concatenate((p_h, p_h, p_h), axis=0)
    style_img = style_img[np.newaxis, ...]
    content_img = content_img[np.newaxis, ...]

    # image normalization
    style_min = style_img.min()
    style_max = style_img.max()
    content_min = content_img.min()
    content_max = content_img.max()
    style_img = (style_img - style_min) / (style_max - style_min)
    content_img = (content_img - content_min) / (content_max - content_min)

    # add noise
    random_seed = random_seed
    if random_seed:
        np.random.seed(random_seed)
    noise = np.random.normal(0, 0, content_img.shape)
    content_img += noise
    content_img = np.clip(content_img, 0, 1)

    # to Tensor
    style_img = torch.Tensor(style_img.astype(float))
    content_img = torch.Tensor(content_img.astype(float))
    style_img = style_img.to(device, torch.float)
    content_img = content_img.to(device, torch.float)


    # ====== RED with style transfer ======
    # specify the model to use for style transfer
    cnn = models.vgg16(weights='VGG16_Weights.DEFAULT').features.to(device).eval()

    # cnn normalization param for model
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone().detach()
    # initialize style transfer
    style_transfer = StyleTransfer(
        cnn,
        normalization_mean=cnn_normalization_mean,
        normalization_std=cnn_normalization_std,
        content_img=content_img,
        style_img=style_img,)

    # call RED solver
    # parser_style_transfer [num_steps, style_weight, content_weight]
    rnst_solver = RNST(
        denoiser=None,
        style_transfer=style_transfer,
        device=device,
        flag_bm3d=True,
        parser_style_transfer=[500, 5000, 1],)

    N_iter, miu, lam, update_num_steps, num_style_step = engine_settings
    output_list, output_bestall = rnst_solver.solve_RNST(
        img_in=input_img,
        img_ref=p_ref,
        N_iter=N_iter,
        show_steps=show_steps,
        miu=miu,
        lam=lam,
        std_noise=std_noise,
        update_num_steps=update_num_steps,
        num_style_step=num_style_step,
        use_hybrit_loss=use_hybrid_loss,
        show_info=show_info,)

    # image denormalization
    content_img = content_img.cpu().numpy()
    style_img = style_img.cpu().numpy()
    content_img = content_img * (content_max - content_min) + content_min
    style_img = style_img * (style_max - style_min) + style_min

    # output image
    output = output_bestall
    img_transfer = output.detach().cpu().numpy().squeeze()
    img_transfer = np.transpose(img_transfer, (1, 2, 0))
    img_transfer = img_transfer[..., 0]
    img_transfer = img_transfer * (content_max - content_min) + content_min


    # ====== evaluation ======
    content_img = (content_img - content_min) / (content_max - content_min)
    img_transfer = (img_transfer - content_min) / (content_max - content_min)
    p_ref = (p_ref - p_ref.min()) / (p_ref.max() - p_ref.min())

    ssim_input = calc_ssim(p_ref, content_img[0, 0])
    ssim_output = calc_ssim(p_ref, img_transfer)
    psnr_input = calc_psnr(p_ref, content_img[0, 0])
    psnr_output = calc_psnr(p_ref, img_transfer)

    return ssim_input, ssim_output, psnr_input, psnr_output


def RNST_freezeguid_engine(
        path_content,
        path_style,
        path_ref,
        idx_pulse,
        idx_slice,
        idx_guid,
        std_noise,
        engine_settings=(20, 0.15, 0.7, 5, 3),  #[N_iter,miu,lam,update_num_steps,num_style_step]
        random_seed=None,
        show_steps=False,
        show_info=False,
        use_hybrid_loss=False,
):
    '''
    An RNST_engine that can indicate the style image specifically.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ====== construct data ======
    # load img
    if idx_slice+1 < 10:
        path_ref = path_ref + '/IMG000' + str(idx_slice+1) + '.dcm'  # start from 1
    else:
        path_ref = path_ref + '/IMG00' + str(idx_slice+1) + '.dcm'
    if idx_guid+1 < 10:
        path_p_h = path_style + '/IMG000' + str(idx_guid+1) + '.dcm'  # start from 1
    else:
        path_p_h = path_style + '/IMG00' + str(idx_guid+1) + '.dcm'
    if idx_pulse != 2:
        path_p_l = path_content + '/vs001_' + str(idx_slice) + '.dcm'  # start from 0
    else:
        path_p_l = path_content + '/vs001_' + str(idx_slice+80) + '.dcm'  # start from 80

    p_h = load_img(path_p_h)  # high field img
    p_l = load_img(path_p_l)  # low field img
    p_ref = load_img(path_ref)  # reference img, registered scan

    # concatenate as RGB images
    p_h = p_h[np.newaxis, ...]
    p_l = p_l[np.newaxis, ...]
    content_img = np.concatenate((p_l, p_l, p_l), axis=0)  # low2high
    style_img = np.concatenate((p_h, p_h, p_h), axis=0)
    style_img = style_img[np.newaxis, ...]
    content_img = content_img[np.newaxis, ...]

    # image normalization
    style_min = style_img.min()
    style_max = style_img.max()
    content_min = content_img.min()
    content_max = content_img.max()
    style_img = (style_img - style_min) / (style_max - style_min)
    content_img = (content_img - content_min) / (content_max - content_min)

    # add noise
    random_seed = random_seed
    if random_seed:
        np.random.seed(random_seed)
    noise = np.random.normal(0, 0, content_img.shape)
    content_img += noise
    content_img = np.clip(content_img, 0, 1)

    # to Tensor
    style_img = torch.Tensor(style_img.astype(float))
    content_img = torch.Tensor(content_img.astype(float))
    style_img = style_img.to(device, torch.float)
    content_img = content_img.to(device, torch.float)


    # ====== RED with style transfer ======
    # specify the model to use for style transfer
    cnn = models.vgg16(weights='VGG16_Weights.DEFAULT').features.to(device).eval()

    # cnn normalization param for model
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone().detach()
    # initialize style transfer
    style_transfer = StyleTransfer(
        cnn,
        normalization_mean=cnn_normalization_mean,
        normalization_std=cnn_normalization_std,
        content_img=content_img,
        style_img=style_img,)

    # call RED solver
    # parser_style_transfer [num_steps, style_weight, content_weight]
    rnst_solver = RNST(
        denoiser=None,
        style_transfer=style_transfer,
        device=device,
        flag_bm3d=True,
        parser_style_transfer=[500, 5000, 1],)

    N_iter, miu, lam, update_num_steps, num_style_step = engine_settings
    output_list, output_bestall = rnst_solver.solve_RNST(
        img_in=input_img,
        img_ref=p_ref,
        N_iter=N_iter,
        show_steps=show_steps,
        miu=miu,
        lam=lam,
        std_noise=std_noise,
        update_num_steps=update_num_steps,
        num_style_step=num_style_step,
        use_hybrit_loss=use_hybrid_loss,
        show_info=show_info,)

    # image denormalization
    content_img = content_img.cpu().numpy()
    style_img = style_img.cpu().numpy()
    content_img = content_img * (content_max - content_min) + content_min
    style_img = style_img * (style_max - style_min) + style_min

    # output image
    output = output_bestall
    img_transfer = output.detach().cpu().numpy().squeeze()
    img_transfer = np.transpose(img_transfer, (1, 2, 0))
    img_transfer = img_transfer[..., 0]
    img_transfer = img_transfer * (content_max - content_min) + content_min

    # ====== evaluation ======
    content_img = (content_img - content_min) / (content_max - content_min)
    img_transfer = (img_transfer - content_min) / (content_max - content_min)
    p_ref = (p_ref - p_ref.min()) / (p_ref.max() - p_ref.min())

    ssim_input = calc_ssim(p_ref, content_img[0, 0])
    ssim_output = calc_ssim(p_ref, img_transfer)
    psnr_input = calc_psnr(p_ref, content_img[0, 0])
    psnr_output = calc_psnr(p_ref, img_transfer)

    return ssim_input, ssim_output, psnr_input, psnr_output
