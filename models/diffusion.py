import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1 #* 确保timessteps是一维张量

    half_dim = embedding_dim // 2 #* 因为embedding包含一般正弦盒一般余弦
    emb = math.log(10000) / (half_dim - 1) #* 根据公式计算频率参数，用于控制正弦和余弦的频率
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb) #* 生成一个频率序列
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :] #* None表示增加一个新维度
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    #? Group Normalization：将每个样本的通道维度划分为多个组，然后在每个组内进行归一化操作
    #? 参数：num_groups-把channel分成几组; num_channels-输入的通道数 \
    #? eps-防止除0; affine-使用可学习的仿射变换参数(尺度和位移)
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    '''
    input:  [batch_size, in_channels, height, width]
    output: [batch_size, in_channels, 2*height, 2*width]
    '''
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv #* bool值：上采样后是否应用卷积操作
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #* 进行上采样，将x的高度和宽度扩大一倍，模式为nearest
        x = torch.nn.functional.interpolate( 
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    '''
    input:  [batch_size, in_channels, height, width]
    output: [batch_size, in_channels, height//2, width//2]
    '''
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv #* bool：下采样后是否应用卷积操作
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2, #* 注意stride是2
                                        padding=0) #* padding是0

    def forward(self, x):
        if self.with_conv: #* 如果使用conv，通过卷积操作来实现downsampling
            pad = (0, 1, 0, 1) #* 自己做asymmetric padding，pad=(left,right,up,down)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else: #* 如果不用conv，直接通过pooling来做downsampling
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    ''' 输入输出形状相同
    input:  [batch_size, in_channels, height, width]
    output: [batch_size, in_channels, height, width]
    structure: x克隆出来一个h, 往下走, 中间增加time embedding, 最后增加res x
    '''
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512): #* time embedding的通道数
        #* 初始化工作
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        #* 定义层
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels) #* 处理time embedding的通道数
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) #* dropout：Dropout概率，用于正则化
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        #* 用于res conn的downsampling类型
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut: #* 使用3x3卷积核来调整通道数
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, 
                                                     kernel_size=3, stride=1, padding=1)
            else: #* 使用1x1卷积核，单纯为了调整通道数
                #? nin：Network In Network
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, 
                                                    kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h) #* swish激活函数
        h = self.conv1(h)

        #* 将time embedding增加到feature map中
        #? [:,:,None,None]用于调整维度，使时间嵌入向量的形状与特征图 h 的形状兼容
        #? [batch_size, out_channels] -> [batch_size, out_channels, 1, 1]
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h) #* swish激活函数
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            #* 判断用于res conn的x的downsampling类型
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h #* residual connection在这里实现


class AttnBlock(nn.Module):
    ''' 输入输出形状相同
    input:  [batch_size, in_channels, height, width]
    output: [batch_size, in_channels, height, width]
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw  w[b,i,j]=sum_c q[b,i,c]k[b,c,j] #* batch matrix multiplication
        w_ = w_ * (int(c)**(-0.5)) #* 对注意力权重进行缩放，缓解点积的数值范围过大问题
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_ #* res conn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #* ch：基准通道数，初始化卷积层的输出通道数; out_ch: 模型的最终输出通道数; ch_mult是通道倍增系数列表，转换为元组
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult) 
        num_res_blocks = config.model.num_res_blocks #* 每个分辨率级别中参差块的数量
        attn_resolutions = config.model.attn_resolutions #* 列表，指示在哪些分辨率级别应用注意力机制
        dropout = config.model.dropout #* 丢弃概率，用于正则化
        in_channels = config.model.in_channels #* 输入通道数 
        resolution = config.data.image_size #* 图像的尺寸，就是一个数，默认是正方形
        resamp_with_conv = config.model.resamp_with_conv #* bool：是否在上下采样时使用卷积
        num_timesteps = config.diffusion.num_diffusion_timesteps #* 扩散过程的时间步数
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4 #* time embedding的通道数是基准通道数的4倍
        self.num_resolutions = len(ch_mult) #* 存在几种分辨率级别
        self.num_res_blocks = num_res_blocks #* 每个分辨率级别中参差块的数量
        self.resolution = resolution #* 图像的尺寸
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, #* 初始卷积层，改变通道数
                                       kernel_size=3, stride=1, padding=1)

        curr_res = resolution #* 图像尺寸，一个数，默认图像是正方形
        in_ch_mult = (1,)+ch_mult #* 通道倍增系数
        self.down = nn.ModuleList()
        block_in = None #* 初始化输入通道数量，在每一个分辨率等级/阶段都不一样
        for i_level in range(self.num_resolutions): #* 遍历downsample的几个分辨率等级
            block = nn.ModuleList() #* 定义当前分辨率等级的参差块
            attn = nn.ModuleList() #* 定义当前分辨率等级的注意力块
            block_in = ch*in_ch_mult[i_level] #* 计算当前分辨率等级的输入通道数
            block_out = ch*ch_mult[i_level] #* 计算当前分辨率等级的输出通道数
            for i_block in range(self.num_res_blocks): #* 每个分辨率等级做x个参差块上去
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: #* 在合适的分辨率位置，增加attention block
                    attn.append(AttnBlock(block_in))
            #* 把这个分辨率等级的component集成到一个down里面
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1: #* 如果不是最后一个分辨率等级，执行downsample
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            #? self.down是一个MoudleList，包含num_resolutions个Module,
            #? 每个Module包含：block*n + (attn*1) + (downsample*1)

        # middle
        #* 只有ResBlock和AttnBlock，他们都不改变数据的形状
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): #* 遍历几个分率等级，注意是reversed
            block = nn.ModuleList() #* 当前分辨率的参差块
            attn = nn.ModuleList() #* 当前分辨率的注意力块
            block_out = ch*ch_mult[i_level] #* 当前分辨率的输出通道数
            skip_in = ch*ch_mult[i_level] #* 用于跳跃连接(skip connection)输入通道数，在每次倍增第一个的位置
            for i_block in range(self.num_res_blocks+1): #* 每个分辨率等级做x个参差块，多了一个，用于上采样后处理
                #? 上采样后处理：看作是和downsample对应位置的res conn就行
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level] #* 对应downsample地方的通道数
                block.append(ResnetBlock(in_channels=block_in+skip_in, #* 融合了downsample阶段的特征
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: #* 在合适的分辨率位置，增加attention block
                    attn.append(AttnBlock(block_in))
            #* 把这个分辨率等级的component集成到一个up里面  
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0: #* 如果不是在第一个分辨率等级，执行upsample
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order #* insert到开头的地方
            #? self.up是一个MoudleList，包含num_resolutions个Module,
            #? 每个Module包含：(upsample*1) + block*(n+1) + (attn*1)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, #* 和conv_in相对应，结束卷积层，改变通道数
                                        kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch) #* 获取time embedding
        temb = self.temb.dense[0](temb) #* 后续处理生成高纬嵌入向量
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        #? 为什么要用hs：因为在后面的upsampling部分需要又skip connection
        hs = [self.conv_in(x)] #* [b, self.ch, h, w]
        for i_level in range(self.num_resolutions): #* 遍历下采样阶段的每个分辨率等级
            for i_block in range(self.num_res_blocks): #* 遍历每个分辨率等级的参差块
                h = self.down[i_level].block[i_block](hs[-1], temb) #* index[-1] 用于获取最新的特征图
                if len(self.down[i_level].attn) > 0: 
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h) #* 每经过一个参差块(和注意力块)，最新的特征图会被增加到hs的末尾
            if i_level != self.num_resolutions-1: #* 如果是最后一个分辨率等级，执行downsampling
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1] #* 提取最新的特征图
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)): #* 注意是reversed，遍历每个分辨率等级
            for i_block in range(self.num_res_blocks+1): #* 遍历每个分辨率等级的(参差块数量+1)
                h = self.up[i_level].block[i_block](
                    #? pop(): 弹出并返回最后一个特征图，kind like stack
                    torch.cat([h, hs.pop()], dim=1), temb) #* 在channel维度cat
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0: #* 如果不是到了最高分辨率等级，就执行upsampling
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
