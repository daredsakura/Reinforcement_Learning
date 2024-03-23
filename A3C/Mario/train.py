import argparse
import os
import torch
from src.env import create_train_env
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import train
import torch.multiprocessing as _mp  # pytorch 多进程
import shutil

os.environ['OMP_NUM_THREADS'] = '1'


# 对世界关卡和训练超参数通过命令行设置
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def main(opt):
    device = torch.device("cuda:0" if opt.use_gpu else "cpu")
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    # 在使用spawn方式启动子进程时，父进程将会在一个全新的Python解释器中启动子进程
    mp = _mp.get_context('spawn')
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(num_states, num_actions).to(device)
    global_model.share_memory()
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
