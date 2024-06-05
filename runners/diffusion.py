import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu


#todo 将tensor转换位unit8格式的图像数据
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


#todo 根据不同的参数生成不同的beta调度数组，返回值是一个[num_diff_ts, ]的beta数组
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad": #* 平方调度（quadratic）
        betas = ( #* 先根号生成均匀分布的序列，再平方
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear": #* 线性调度     
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const": #* 常数调度
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace( #* 跟选取的β_start β_end就没有关系了
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,) #* 确保beta序列的形状和timesteps一致
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = device

        self.model_var_type = config.model.var_type #* 指定方差类型
        betas = get_beta_schedule( #* 根据参数获取beta数组
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device) #* 转为tensor并移动到device
        self.num_timesteps = betas.shape[0] 

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0) #* α的累计乘积
        alphas_cumprod_prev = torch.cat( #* 前一时间步的α累计乘积，并在开头增加一个'1'
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = ( #! 后验方差，用于扩散模型的采样过程
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        #* 设置模型的对数方差logvar，用于后续的模型计算和采样过程
        if self.model_var_type == "fixedlarge": #* 使用β的对数值作为方差
            #? .log()用于计算自然对数ln
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall": #* 使用后验方差的对数值作为方差
            #? min=1e-20：确保最小值不小于 1e-20 以避免数值不稳定
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model) #* 用于多GPU并行训练

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training: #* 如果是恢复训练
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth")) 
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time() 
            data_time = 0 #* 训练一个epoch所需要的时间
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0) #* batch size
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x) #* 对数据进行预处理
                e = torch.randn_like(x) #* 与输入数据相同的随机噪声
                b = self.betas #! 暂时不知道是干嘛的，感觉多此一举

                # antithetic sampling
                t = torch.randint( #* 生成随机时间步
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n] #* 反对称处理
                loss = loss_registry[config.model.type](model, x, t, e, b) #* 根据cfg选择loss fn并计算loss

                tb_logger.add_scalar("loss", loss, global_step=step) #* 记录损失到TensorBoard

                logging.info( #* 记录训练日志
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad() 
                loss.backward()

                try: #* 在进行参数更新之前，执行梯度裁剪，以防止梯度爆炸
                    #? clip_grad_norm_(parameters, max_norm): 第一个arg为要限制的参数，第二个arg为最大阈值
                    torch.nn.utils.clip_grad_norm_( #* 限制每个参数梯度的翻书
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                #* 保存模型的check point
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [ #* 获取当前状态
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save( #* 保存状态到历史ckpt路径
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth")) #* 保存状态到默认路径

                data_start = time.time() #* 更心start time，用于计算下一个batch的训练用时

    #todo 主采样函数，根据参数调用具体的采样方法
    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained: #* 如果不使用预训练权重，则加载最近一次训练的模型权重
            #? getattr(object, attribute, default): 输出object的attribute，如果没有输出default
            if getattr(self.config.sampling, "ckpt_id", None) is None: #* 如果cfg里没有ckpt
                #? torch.load(path, map_location)：加载torch.save的数据，参数位路径和映射的设备
                states = torch.load( #* 加载默认check point
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else: #* 如果cfg里存在ckpt
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = model.to(self.device)
            model = torch.nn.DataParallel(model) #* 用于在多GPU上并行运算
            model.load_state_dict(states[0], strict=True) #* 加载模型权重

            #! 暂时不知道是干什么的，先跳过了
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else: #* 使用预训练权重
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            #* 选择与训练模型
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device)) #* 加载权重
            model.to(self.device) #* 转移设备
            model = torch.nn.DataParallel(model) #* 多GPU并行运算

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    #todo 用于生成图像样本以进行 FID（Fréchet Inception Distance）评估
    def sample_fid(self, model):
        '''大概就是生成了一堆image, 然后保存起来了'''
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000 #* 总共要生成的样本数量
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size #* 计算生成需要的轮数

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x) #* 将模型内部的图像表示转换为实际的图像表示

                for i in range(n): #* 遍历保存生成的image
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    #todo 用于生成图像序列，展示扩散过程中的中间结果
    def sample_sequence(self, model):
        '''生成了[num_step, 8, c, h, w]的图像'''
        config = self.config

        x = torch.randn( 
            8, #* batch size，这里生成八张image
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            #* last=False 表示不只返回最后一步的结果，而是返回每一步的中间结果
            _, x = self.sample_image(x, model, last=False) #* x: [num_step, 8, channel, img_size, img_size]

        x = [inverse_data_transform(config, y) for y in x] #* 将模型内部的图像表示转换为实际的图像表示

        for i in range(len(x)): #* 遍历num_step
            for j in range(x[i].size(0)): #* 遍历batch_size
                tvu.save_image( #* 存储图像
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    #todo 用于在两个潜在变量之间进行插值，生成平滑过渡的图像（？）
    def sample_interpolation(self, model):
        config = self.config

        #todo 定义球面线性插值 (slerp) 函数
        def slerp(z1, z2, alpha):
            '''用于在两个潜在向量z1和z2之间进行球面线性插值, 
            生成一系列从 z1 平滑过渡到 z2 的中间向量
            '''
            #* 计算向量z1 z2之间的夹角
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                #* alpha是一个权重系数，决定了插值的位置
                #* α=0, 返回z1; α=1, 返回z2; α=[0, 1], 返回z1 z2的中间向量;
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device) #* alpha是一个权重系数，决定了插值的位置
        z_ = [] #* 生成插值潜在向量
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0) #* 将插值潜在向量拼接成一个张量 
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8): #* 将插值向量 x 分成批次，每次处理8个向量
                xs.append(self.sample_image(x[i : i + 8], model))
        
        x = inverse_data_transform(config, torch.cat(xs, dim=0)) #* 将模型内部的图像表示转换为实际的图像表示
        for i in range(x.size(0)): #* 遍历存储图像
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    #todo 通用采样函数：根据不同的方法(generalized / ddpm_noisy)生成图像
    def sample_image(self, x, model, last=True):
        '''去噪真正进行的地方 (指调用了去噪函数)'''
        
        #? skip 参数用于控制采样过程中跳过的时间步数; 
        #? 它的主要作用是减少采样步骤，从而加速采样过程，而不会显著影响生成图像的质量
        try: #* 尝试获取skip参数
            skip = self.args.skip
        except Exception:
            skip = 1 #* 默认skip为1

        #* 不同的sample type，调用的denosing函数不同
        if self.args.sample_type == "generalized": 
            if self.args.skip_type == "uniform": #* 均匀跳步
                #? 前者是扩散过程的总timestep，后者是用户指定的timestep
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad": #* 二次跳步（平方跳步）
                seq = (
                    np.linspace( #* 先根号生成均匀分布的序列，再平方
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            
            #* general 就调用 general的enoising函数
            from functions.denoising import generalized_steps
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            #* skip部分完全同上
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            
            #* ddpm 就调用 ddpm的denoising函数
            from functions.denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        
        if last:
            x = x[0][-1] #! x的形状：yet to be figure out, 和xs相同; 推测为[n, b, c, h, w]
        return x

    def test(self):
        pass
