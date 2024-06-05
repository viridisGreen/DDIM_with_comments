import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor, #* 原始图像张量，[batch_size, channels, height, width]
                          t: torch.LongTensor, #* 时间步长张量，[batch_size]
                          e: torch.Tensor, #* 真实噪声张量，[batch_size, channels, height, width]
                          b: torch.Tensor, #* 扩散过程中噪声因子张量，[num_timesteps]
                          keepdim=False): #* 返回误差list还是误差mean
    #? cumprod：计算张量沿指定维度的累乘; 返回一个与输入形状相同的张量，每个元素是从该维度开始到当前位置的累乘
    #? index_select(dim, index): 在指定维度上，根据索引张量选取元素
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1) #* noise factor, 用于控制每个时间步的噪声强度
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt() #* 不同时间步上加噪的图像，就是加噪的过程
    output = model(x, t.float()) #* 预测噪声（noise residual）
    if keepdim: #* 返回每个样本的总误差，形状为 [batch_size]
        return (e - output).square().sum(dim=(1, 2, 3)) #* 平方误差
    else: #* 计算所有样本的平均误差，返回标量
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
