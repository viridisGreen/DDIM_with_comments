import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    #todo 增加命令行参数
    parser.add_argument( 
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument( #* 用于文档记录的字符串，将作为日志文件夹的名称 
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument( #* 设置日志的详细程度
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true") #* 是否计算FID
    parser.add_argument("--interpolation", action="store_true") #* 是否进行图像插值操作，评估生成模型的潜在空间结构
    parser.add_argument( #* 是否从上次中断的地方继续训练
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument( #* 指定保存生成样本图像的文件夹名称
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument( #* 是否启用无交互模式
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument( #! 指定采样方法，可选generalized或ddpm_noisy
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument( #! 指定跳过类型，可选uniform或quadratic
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument( #! 控制 sigma 方差的 eta 参数
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true") #! 是否按序列处理

    args = parser.parse_args() #* 解析命令行参数，并存在args中
    args.log_path = os.path.join(args.exp, "logs", args.doc) #* 设置日志路径

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f) #* config是字典
    new_config = dict2namespace(config) #* 将 字典对象 转换为 命名空间对象
    #?TIP: 命名空间对象：同字典，但可以通过'.'运算符去访问对象，而不需要'['xxx']'，更方便

    tb_path = os.path.join(args.exp, "tensorboard", args.doc) #* 设置tensor board日志路径

    #TODO 根据不同的命令行参数设置日志目录和TensorBoard日志目录
    #todo 非测试模式 or 采样模式下的处理
    if not args.test and not args.sample: #* 不是在测试 or 采样模式下
        if not args.resume_training: #* 不是恢复训练的状态（应该是指续训）
            #* 判断日志目录是否存在，不存在就mkdir
            if os.path.exists(args.log_path):
                overwrite = False #* 一个flag，用于标志在目录存在的情况下是否覆写
                if args.ni: #* 如果启用了无交互模式
                    overwrite = True #* 启用无交互模式默认一定覆写
                else:
                    #* 没有启用无交互模式，根据用户输入决定是否覆写
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite: #* 如果覆写，删除现有目录并重新创建
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else: #* 如果目录存在又不覆写，那就直接退出程序
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else: 
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False) #* 将new_cfg保存到f中，以yaml格式

        #* 初始化一个TensorBoard日志记录器，并赋值
        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        
        # setup logger
        level = getattr(logging, args.verbose.upper(), None) #* 确定日志级别（从logging中获取）
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        #* 创建两个日志处理器，分别用于控制台和写入文件
        handler1 = logging.StreamHandler() #* 将日志输出到控制台
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt")) #* 将日志输出到文件
        formatter = logging.Formatter( #* 日志各式器formatter，定义日志的输出格式
            #* 日志级别 - 文件名 - 时间戳 - 消息内容
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter) #* 设置日志处理器输出格式
        handler2.setFormatter(formatter)
        logger = logging.getLogger() #* 创建默认日志记录器
        logger.addHandler(handler1) #* 将handler1增加到日志记录器中，日志记录器会使用handler1将日志输出到控制台
        logger.addHandler(handler2) #* 将handler2增加到日志记录器中，日志记录器会使用handler2将日志输出到文件
        logger.setLevel(level) #* 设置日志记录器的级别，这决定了日志记录器将记录哪些级别的日志

    #todo 测试模式 or 采样模式下的处理
    else:
        level = getattr(logging, args.verbose.upper(), None) #* 确定日志级别（从logging中获取）
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler() #* 创建日志处理器，输出到控制台
        formatter = logging.Formatter( #* 日志各式器formatter，定义日志的输出格式
            #* 日志级别 - 文件名 - 时间戳 - 消息内容
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter) #* 设置日志处理器输出格式
        logger = logging.getLogger() #* 创建默认日志记录器
        logger.addHandler(handler1) #* 将handler1增加到日志记录器中，日志记录器会使用handler1将日志输出到控制台
        logger.setLevel(level) #* 设置日志记录器的级别，这决定了日志记录器将记录哪些级别的日志

        #todo 如果是在采样模式下：样本处理逻辑
        #* 简单来说就是增加了存储sample的folder的操作
        if args.sample: 
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True) #* 创建保存样本的目录
            args.image_folder = os.path.join( #* 设置样本文件夹的路径
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder): #* 如果不存在img_folder，则创建
                os.makedirs(args.image_folder)
            else:
                #? fid和interpolation都是布尔类型，决定是否计算fid或对图像进行插值操作
                if not (args.fid or args.interpolation): #* 如果启用了其中一项，则跳过下面的部分
                    overwrite = False #* 一个flag，用于标志在目录存在的情况下是否覆写
                    if args.ni: #* 如果启用了无交互模式
                        overwrite = True #* 启用无交互模式默认一定覆写
                    else: #* 如果没有启用无交互模式，根据用户输入判断是否覆写
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite: #* 如果覆写，删除目录并重新创建
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else: #* 如果目录存在又不覆写，直接退出程序
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed) #* 设置pytorch全局随机种子，从命令行参数中获取
    np.random.seed(args.seed) #* 设置numpy全局随机种子，从命令行参数中获取
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) #* 设置gpu设备的随机种子

    #* 启用 cuDNN 自动调优; 这样可以在每次运行时选择最优的算法来加速卷积运算，提高训练速度
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace() #* 创建Namespace对象
    for key, value in config.items(): #* 遍历字典
        if isinstance(value, dict): #* 处理嵌套字典
            new_value = dict2namespace(value) #* 递归调用dict2namespace函数
        else:
            new_value = value
        setattr(namespace, key, new_value) #* 将key和value关联
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main()) #* 在main执行完毕后退出程序，并将main的返回值作为退出状态码返回给系统
