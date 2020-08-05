#!/bin/bash
#SBATCH -J 5wslsk6                   # 作业名为 test
#SBATCH -p normal                    # 作业提交的分区为 cpu
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH -n 9              # 单个节点使用 1 块 GPU 卡
#SBATCH -t 1:59:00                # 任务运行的最长时间为 1 小时
#SBATCH -o 5wsls.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -e 5wsls.err

module load anaconda3     # 添加 anaconda/3-5.0.0.1 模块

python wsls_efficient.py                    # 执行命令