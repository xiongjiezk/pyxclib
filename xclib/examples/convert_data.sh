# !/usr/bin/env bash
#
# Copyright @2022 AI, ZHIHU Inc. (zhihu.com)
#
# @author: xiongjie <xiongjie@zhihu.com>
# @date: 2022/08/07
#

dataset=$1
from_dir="/home/xiongjie/work/data/rank_xmc/data/${dataset}/raw_data"
to_dir="/home/xiongjie/work/data/rank_xmc/code/Tree_Extreme_Classifiers/Sandbox/Data/${dataset}"

mkdir -p "${to_dir}"

trn_in_file="${from_dir}/train_%s.txt"
tst_in_file="${from_dir}/test_%s.txt"
trn_out_file="${to_dir}/train_svm.txt"
trn_out_xf="${to_dir}/trn_X_Xf.txt"
trn_out_x_y="${to_dir}/trn_X_Y.txt"
tst_out_file="${to_dir}/test_svm.txt"
tst_out_xf="${to_dir}/tst_X_Xf.txt"
tst_out_x_y="${to_dir}/tst_X_Y.txt"

python sparse_bow_features_from_raw_data.py $trn_in_file $tst_in_file $trn_out_file $tst_out_file
perl convert_format.pl $trn_out_file $trn_out_xf $trn_out_x_y
perl convert_format.pl $tst_out_file $tst_out_xf $tst_out_x_y




