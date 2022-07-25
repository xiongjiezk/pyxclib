# !/usr/bin/env bash
#
# Copyright @2022 AI, ZHIHU Inc. (zhihu.com)
#
# @author: xiongjie <xiongjie@zhihu.com>
# @date: 2022/08/07
#

dataset=$1
data_dir="/home/xiongjie/work/data/rank_xmc/code/Tree_Extreme_Classifiers/Sandbox/Data/${dataset}"
result_dir="/home/xiongjie/work/data/rank_xmc/code/Tree_Extreme_Classifiers/Sandbox/Results/${dataset}"
target_file="${to_dir}/tst_X_Y.txt"
pred_file="${result_dir}/score_mat.txt"
python evaluate.py $target_file $pred_file
