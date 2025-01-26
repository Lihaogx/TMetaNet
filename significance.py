import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from tabulate import tabulate
from glob import glob
import pandas as pd
from scipy import stats

def calculate_significance(values1, values2):
    """
    计算两组数据的显著性差异
    
    Args:
        values1: 第一组实验的指标值列表
        values2: 第二组实验的指标值列表
    
    Returns:
        mean_diff: 平均值差异 (values1_mean - values2_mean)
        p_value: t检验的p值
        significant: 是否显著 (p < 0.05)
    """
    if not values1 or not values2:
        return None, None, False
    
    mean1, mean2 = np.mean(values1), np.mean(values2)
    mean_diff = mean1 - mean2
    _, p_value = stats.ttest_ind(values1, values2)
    return mean_diff, p_value, p_value < 0.05

def process_experiment_results(exp1_path, exp2_path):
    """
    处理两个实验结果并比较显著性
    
    Args:
        exp1_path: 第一个实验的完整路径
        exp2_path: 第二个实验的完整路径
    """
    if not os.path.exists(exp1_path) or not os.path.exists(exp2_path):
        return ""
    
    metrics = {
        'accuracy': [], 'mrr': [], 'rck1': [], 'rck3': [], 
        'rck10': [], 'micro_auc': [], 'macro_auc': []
    }
    
    # 分别存储两个实验的指标
    exp1_metrics = {k: [] for k in metrics.keys()}
    exp2_metrics = {k: [] for k in metrics.keys()}
    
    # 获取实验名称
    exp1_name = os.path.basename(exp1_path)
    exp2_name = os.path.basename(exp2_path)
    
    # 处理第一个实验
    run_dirs = [d for d in os.listdir(exp1_path) if os.path.isdir(os.path.join(exp1_path, d))]
    for run_dir in run_dirs:
        run_path = os.path.join(exp1_path, run_dir)
        for metric in metrics.keys():
            metric_path = os.path.join(run_path, f'test_{metric}')
            if os.path.exists(metric_path):
                event_files = [f for f in os.listdir(metric_path) if f.startswith('events.out.tfevents.')]
                if event_files:
                    ea = event_accumulator.EventAccumulator(os.path.join(metric_path, event_files[0]))
                    ea.Reload()
                    test_values = [s.value for s in ea.Scalars('test')]
                    if test_values:
                        exp1_metrics[metric].append(np.mean(test_values))
    
    # 处理第二个实验
    run_dirs = [d for d in os.listdir(exp2_path) if os.path.isdir(os.path.join(exp2_path, d))]
    for run_dir in run_dirs:
        run_path = os.path.join(exp2_path, run_dir)
        for metric in metrics.keys():
            metric_path = os.path.join(run_path, f'test_{metric}')
            if os.path.exists(metric_path):
                event_files = [f for f in os.listdir(metric_path) if f.startswith('events.out.tfevents.')]
                if event_files:
                    ea = event_accumulator.EventAccumulator(os.path.join(metric_path, event_files[0]))
                    ea.Reload()
                    test_values = [s.value for s in ea.Scalars('test')]
                    if test_values:
                        exp2_metrics[metric].append(np.mean(test_values))
    
    # 计算显著性并生成结果
    results = []
    for metric in metrics.keys():
        mean_diff, p_value, is_significant = calculate_significance(
            exp1_metrics[metric], exp2_metrics[metric]
        )
        
        if mean_diff is not None:
            exp1_mean = np.mean(exp1_metrics[metric])
            exp1_std = np.std(exp1_metrics[metric])
            exp2_mean = np.mean(exp2_metrics[metric])
            exp2_std = np.std(exp2_metrics[metric])
            
            better_exp = exp1_name if mean_diff > 0 else exp2_name
            significance_mark = '*' if is_significant else ''
            
            results.append([
                metric.upper(),
                f"{exp1_mean:.4f}±{exp1_std:.4f}",
                f"{exp2_mean:.4f}±{exp2_std:.4f}",
                f"{abs(mean_diff):.4f}",
                f"{p_value:.4f}",
                better_exp + significance_mark
            ])
    
    # 生成表格
    headers = ["指标", exp1_name, exp2_name, "差异绝对值", "p值", "更好的实验"]
    
    # 保存CSV
    parent_dir = os.path.dirname(os.path.dirname(exp1_path))  # 获取父目录
    csv_filename = f"comparison_{exp1_name}_vs_{exp2_name}.csv"
    csv_path = os.path.join(parent_dir, csv_filename)
    df = pd.DataFrame(results, columns=headers)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return tabulate(results, headers=headers, tablefmt="grid")

def read_tfevents_file(file_path):
    """
    读取并打印tfevents文件中的内容
    
    Args:
        file_path: tfevents文件的完整路径
    """
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    
    # 打印所有可用的标签
    if not ea.Tags():
        print(f"文件地址: {file_path}")
    
    # 获取并打印'test'标签下的所有值
    if 'test' not in ea.Tags().get('scalars', []):
        print(f"文件地址: {file_path}")
        # test_events = ea.Scalars('test')
        # print("\n测试数据:")
        # for event in test_events:
        #     print(f"Step: {event.step}, Value: {event.value}, Wall Time: {event.wall_time}")

# 使用示例
if __name__ == "__main__":
    exp1_path = "/home/lh/Dowzag_2.0/exp_sh/exp77_wingnn_topo/results/uci_topo"  # 替换为实际的实验路径
    exp2_path = "/home/lh/Dowzag_2.0/exp_sh/exp78_wingnn_baseline/results/uci"  # 替换为实际的实验路径
    print(f"\n比较实验:")
    print(f"实验1: {exp1_path}")
    print(f"实验2: {exp2_path}")
    print("-" * 80)
    result_table = process_experiment_results(exp1_path, exp2_path)
    print(result_table)
    print("-" * 80)

    # file_path = "/home/lh/DowZag/exp_sh/alpha/topo_off/results/alpha_topo_off_gru/3302/events.out.tfevents.1735308312.lh-A6000-2.3191771.72"
    # read_tfevents_file(file_path)
