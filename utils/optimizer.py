import numpy as np
import torch
import bm3d
import copy

from utils.evaluation import *



def gradient_decsent_list_upate(
        self,
        device,
        p_ref,
        x_in,
        x_t_list,
        x_d,
        miu,
        lam,
        update_num_steps,
        num_style_step,
        hybrit_output_best,
        use_hybrit_loss=True,
        show_info=True,
):
    '''
    Optimizer function.
    self: RED_ST self.
    p_ref: reference img for evaluation.
    x_in: input img.
    x_t_list: a list of x_t, contains 'num_style_step' orientations. 'num_style_step' is the num of orientations to try.
    x_d: denoised img.
    miu: optimizer factor.
    lam: optimizer factor.
    update_num_steps: factor for the step size along one orientation.
    num_style_step: num of orientations to search for.
    hybrit_output_best: best hybrit output loss.
    use_hybrit_loss: flag, whether to use hybrit loss.
    '''

    p_ref = (p_ref - p_ref.min()) / (p_ref.max() - p_ref.min())

    for i in range(update_num_steps):

        for j in range(num_style_step):
            x_t = x_t_list[j].detach()
            miu_try = (i + 1) * miu
            image_to_compare = x_in - miu_try * (x_in - x_t + lam * (x_in - x_d))

            # normalization
            image_to_test = image_to_compare.detach()
            image_to_test = torch.Tensor(image_to_test).to(device, torch.float)
            image_to_test = (image_to_test - image_to_test.min()) / (image_to_test.max() - image_to_test.min())

            # cal_style_score (range 8-12)
            _, style_loss_output = self.style_transfer.solve(
                image_to_test,
                num_steps=1,
                style_weight=self.style_weight,
                content_weight=self.content_weight,
                show_steps=False,)

            if use_hybrit_loss:
                # cal_ref_score
                image_to_test = image_to_compare.detach().cpu().numpy().squeeze()
                image_to_test = np.transpose(image_to_test, (1, 2, 0))
                image_to_test = image_to_test[..., 0]
                ssim_output_n = calc_ssim(p_ref, image_to_test)  # (range 0.7-0.8)

                hybrit_loss_output = 10 * ssim_output_n + style_loss_output.item()
            else:
                hybrit_loss_output = style_loss_output.item()

            if hybrit_loss_output < hybrit_output_best:
                hybrit_output_best = hybrit_loss_output
                x_output = image_to_compare.clone()
                if show_info:
                    print('* Best hybrit update: -Lev-', i + 1, '-num-', j + 1, '-Value-', hybrit_output_best)

    if show_info:
        print('** Best hybrit Loss:', hybrit_output_best)

    return x_output, hybrit_output_best


class RNST:
    '''
    RNST (Regularization by Neural Style Transfer).
    '''

    def __init__(
            self,
            denoiser,
            style_transfer,
            device,
            flag_bm3d=True,
            parser_style_transfer=(1000, 6000, 1),):

        self.denoiser = denoiser
        self.style_transfer = style_transfer
        self.device = device
        self.flag_bm3d = flag_bm3d
        self.num_steps, self.style_weight, self.content_weight = parser_style_transfer

    def solve_RNST(
            self,
            img_in,
            img_ref,
            N_iter=10,
            show_steps=True,
            miu=0.02,
            lam=0.7,
            std_noise=20/255.,
            update_num_steps=1,
            num_style_step=3,
            use_hybrit_loss=True,
            show_info=True,):
        '''
        solve_RNST solving function
        :param img_in: input img, torch.Tensor, shape [N,3(C),H,W].
        :param img_ref: reference img, np array, shape [H,W].
        :param N_iter: number of iteration of the RED.
        :param show_steps: flag, whether to show step in style transfer engine.
        :param miu: factor for inner gradient update optimizer.
        :param lam: factor for inner gradient update optimizer.
        :param std_noise: noise level for denoiser.
        :param update_num_steps: factor for the step size along one orientation.
        :param num_style_step: style_transfer_reference_number, number of orientations to search for.
        :param show_info: flag, whether to show debugging info.
        '''

        x_in = img_in  # [N,C,H,W], [1,3,512,512]
        num_style_step = num_style_step  # style_transfer_reference_number, number of orientations to search for
        x_final_out_list = []  # style_transfer_reference_list

        # iter
        loss_bestall = 1000
        idx_bestall = -1
        for i in range(N_iter):
            if show_info:
                print()
                print('========= Iter %d =========' % (i))
            # normalization
            x_in = (x_in - x_in.min()) / (x_in.max() - x_in.min())
            x_in_c1 = x_in.clone().detach()
            x_t_list = []

            # image denoising
            if self.flag_bm3d and i == 0:
                x_in_c1 = x_in_c1.detach().cpu().numpy()  # [N,C,H,W]
                x_d = x_in_c1
                for idx_ch in range(x_in.shape[1]):
                    x_d[0, idx_ch, ...] = bm3d.bm3d(x_in_c1[0, idx_ch],
                                                    sigma_psd=std_noise,  # std of the noise
                                                    stage_arg=bm3d.BM3DStages.ALL_STAGES)
                x_d = torch.Tensor(x_d).to(self.device, torch.float)
            else:
                x_d = x_in_c1.detach()
            # normalization
            x_d = (x_d - x_d.min()) / (x_d.max() - x_d.min())

            # style_transfer_d_input
            for j in range(num_style_step):
                x_t, _ = self.style_transfer.solve(x_d,
                                                   num_steps=self.num_steps + 100 * j,
                                                   style_weight=self.style_weight,
                                                   content_weight=self.content_weight,
                                                   show_steps=show_steps)
                x_t_c3 = x_t.clone().detach()
                x_t_list.append(x_t_c3)

            # update rule
            # x_t - style_transter_image
            # x_d - denoising_image
            # x_in - original_image
            hybrit_loss_best = 1000000
            x_out, hybrit_output = gradient_decsent_list_upate(
                self,
                self.device,
                img_ref,
                x_in,
                x_t_list,
                x_d,
                miu,
                lam,
                update_num_steps,
                num_style_step,
                hybrit_loss_best,
                use_hybrit_loss,
                show_info=show_info,)
            x_in = x_out

            # record the best performance
            if hybrit_output < hybrit_loss_best:
                x_final_out = x_out.clone()
                hybrit_loss_best = copy.copy(hybrit_output)
                if show_info:
                    print('** Loss Best updated:', hybrit_loss_best)
                x_final_out_list.append(x_final_out)

            # record the best performance over all iterations
            if hybrit_output < loss_bestall:
                loss_bestall = copy.copy(hybrit_output)
                idx_bestall = i
                if show_info:
                    print('*** Overall Loss Best updated:', hybrit_loss_best)

        return x_final_out_list, x_final_out_list[idx_bestall]