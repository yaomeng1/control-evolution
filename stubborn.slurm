#!/bin/bash
#SBATCH -J c5stubbornk4                  # 作业名为 test
#SBATCH -p normal                    # 作业提交的分区为 cpu
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH -n 9              # 单个节点使用 1 块 GPU 卡
#SBATCH -t 50:59:00                # 任务运行的最长时间为 1 小时
#SBATCH -o c5stubbornk4.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -e c5stubbornk4.err

module load anaconda3     # 添加 anaconda/3-5.0.0.1 模块

python stubborn_sf.py                    # 执行命令