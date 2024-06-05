import torch


#todo 计算噪声比例因子alpha
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    '''
    parameters: 
    x:      初始的噪声张量，[batch_size, channels, height, width]
    seq:    时间步长序列，通常是一个从 num_timesteps 到 0 的递减序列
    model:  噪声估计模型
    b:      扩散过程中的噪声因子张量，形状为 [num_timesteps]; 就是beta
    kwargs: 其他可选参数, 如 eta 
    '''
    with torch.no_grad():
        n = x.size(0) #* batch size
        seq_next = [-1] + list(seq[:-1]) #* 在开头加个-1，这样相同的下标，表示的就是下一个时间步了，很巧妙的方法
        x0_preds = [] #* 用于保存每一步的原始图像预测
        xs = [x] #* 用于保存每一步的图像，从初始噪声图像 x 开始
        for i, j in zip(reversed(seq), reversed(seq_next)):
            #? zip: 用于将两个序列逐元素配对，创建一个由元组组成的迭代器
            #* i, j: 当前时间步，下一时间步
            t = (torch.ones(n) * i).to(x.device) #* 当前时间步张量，[batch_size]
            next_t = (torch.ones(n) * j).to(x.device) #* 下一时间步张量，[batch_size]
            at = compute_alpha(b, t.long()) #* 当前时间步长的噪声比例因子 alpha
            at_next = compute_alpha(b, next_t.long()) #* 下一时间步长的噪声比例因子 alpha
            xt = xs[-1].to('cuda') #* 当前步骤的图像张量
            et = model(xt, t) #* 模型预测噪声
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() #* 当前步骤预测的原始图像，通过去噪公式计算
            x0_preds.append(x0_t.to('cpu')) #* 保存预测的原始图像
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et #* 下一个时间步长生成的图像
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
