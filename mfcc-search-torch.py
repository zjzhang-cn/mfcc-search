import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import warnings
import time

# 过滤 torchaudio.load 的弃用警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

# 全局变量用于统计数据传输
data_transfer_stats = {
    'count': 0,
    'total_bytes': 0,
    'total_time': 0,
    'transfers': []
}

# 设置中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 黑体或微软雅黑
#plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def transfer_to_device(tensor: torch.Tensor, device: torch.device, name: str = ""):
    """将张量传输到指定设备并记录统计信息"""
    if tensor.device == device:
        return tensor
    
    start_time = time.time()
    result = tensor.to(device)
    transfer_time = time.time() - start_time
    
    # 计算数据大小（字节）
    size_bytes = tensor.element_size() * tensor.nelement()
    
    data_transfer_stats['count'] += 1
    data_transfer_stats['total_bytes'] += size_bytes
    data_transfer_stats['total_time'] += transfer_time
    data_transfer_stats['transfers'].append({
        'name': name,
        'size_mb': size_bytes / (1024**2),
        'time': transfer_time,
        'from': str(tensor.device),
        'to': str(device)
    })
    
    return result


def load_mono_audio(path: str, device: torch.device):
    """读取音频并转单声道。"""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return transfer_to_device(waveform, device, f"音频加载: {path}"), sr

# 计算 MFCC 特征
def compute_mfcc(waveform: torch.Tensor, sr: int, n_mfcc: int, n_fft: int, hop_length: int, win_length: int):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_fft,
                   "hop_length": hop_length,
                   "win_length": win_length},
    )
    # 将transform传输到设备
    transform = transform.to(waveform.device)
    mfcc = transform(waveform)  # (channel, n_mfcc, time)
    return mfcc.mean(dim=0)  # (n_mfcc, time)

# 标准化 MFCC 特征, 每个系数维度独立归一化
def normalize(mfcc: torch.Tensor):
    mean = mfcc.mean(dim=1, keepdim=True)
    std = mfcc.std(dim=1, keepdim=True) + 1e-8
    return (mfcc - mean) / std

# 计算统计特征：均值、标准差、最大值、最小值
def stats_features(mfcc: torch.Tensor):
    return torch.cat(
        [
            mfcc.mean(dim=1),
            mfcc.std(dim=1),
            mfcc.max(dim=1).values,
            mfcc.min(dim=1).values,
        ]
    )


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def perform_matching(target_mfcc, source_mfcc, hop_length, sr):
    """执行一次完整的匹配过程并返回结果"""
    # 整体匹配：直接展平后计算余弦相似度
    min_time = min(target_mfcc.shape[1], source_mfcc.shape[1])
    # 计算整体相似度，只比较头部相同长度
    target_flat = target_mfcc[:, :min_time].reshape(-1)
    source_flat = source_mfcc[:, :min_time].reshape(-1)
    overall_similarity = cosine_sim(target_flat, source_flat)

    # 滑动窗口匹配
    query_frames = source_mfcc.shape[1]
    target_frames = target_mfcc.shape[1]

    if query_frames > target_frames:
        target_mfcc, source_mfcc = source_mfcc, target_mfcc
        query_frames = source_mfcc.shape[1]
        target_frames = target_mfcc.shape[1]

    query_norm = normalize(source_mfcc)
    query_flat = query_norm.reshape(-1)
    query_stats = stats_features(source_mfcc)

    best_similarity = -1.0
    best_position = 0
    similarities = []

    for i in range(target_frames - query_frames + 1):
        window = target_mfcc[:, i:i + query_frames]

        window_norm = normalize(window)
        window_flat = window_norm.reshape(-1)
        window_stats = stats_features(window)

        # 计算两部分相似度并加权
        sim_norm = cosine_sim(window_flat, query_flat)
        sim_stats = cosine_sim(window_stats, query_stats)
        sim = 0.7 * sim_norm + 0.3 * sim_stats
        similarities.append(sim)

        if sim > best_similarity:
            best_similarity = sim
            best_position = i

    best_time = best_position * hop_length / sr
    query_duration = query_frames * hop_length / sr

    return {
        'overall_similarity': overall_similarity,
        'best_similarity': best_similarity,
        'best_position': best_position,
        'best_time': best_time,
        'query_duration': query_duration,
        'similarities': similarities,
        'query_frames': query_frames,
        'target_frames': target_frames
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频MFCC特征提取和相似度计算 (PyTorch)')
    parser.add_argument('-t', '--target', required=True, help='目标音频文件路径')
    parser.add_argument('-s', '--source', required=True, help='查询音频文件路径')
    parser.add_argument('-n', '--iterations', type=int, default=10, help='比对次数 (默认: 10)')
    parser.add_argument('-d', '--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='计算设备 (auto: 自动选择, cuda: GPU加速, cpu: CPU计算, 默认: auto)')
    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("警告: CUDA 不可用，将使用 CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 参数配置
    sr = 16000
    n_mfcc = 40
    target_file = args.target
    source_file = args.source
    n_fft = 512  # 25ms
    hop_length = int(sr * 0.01)  # 10ms
    win_length = int(sr * 0.025)  # 25ms
    iterations = args.iterations

    print(f"\n=== 开始加载音频 ===")
    load_start = time.time()
    target_waveform, target_sr = load_mono_audio(target_file, device)
    source_waveform, source_sr = load_mono_audio(source_file, device)
    load_time = time.time() - load_start
    print(f"音频加载时间: {load_time:.4f} 秒")

    # 重采样到目标采样率
    print(f"\n=== 开始重采样 ===")
    resample_start = time.time()
    if target_sr != sr:
        resampler = torchaudio.transforms.Resample(target_sr, sr).to(device)
        target_waveform = resampler(target_waveform)
    if source_sr != sr:
        resampler = torchaudio.transforms.Resample(source_sr, sr).to(device)
        source_waveform = resampler(source_waveform)
    resample_time = time.time() - resample_start
    print(f"重采样时间: {resample_time:.4f} 秒")
    
    # 如果使用GPU，同步以确保所有操作完成
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"\n=== 开始提取 MFCC 特征 ===")
    mfcc_start = time.time()
    target_mfcc = compute_mfcc(
        target_waveform, sr, n_mfcc, n_fft, hop_length, win_length)
    source_mfcc = compute_mfcc(
        source_waveform, sr, n_mfcc, n_fft, hop_length, win_length)
    mfcc_time = time.time() - mfcc_start
    print(f"MFCC 特征提取时间: {mfcc_time:.4f} 秒")

    # 执行多次比对并计算速度
    print(f"\n=== 开始执行 {iterations} 次比对 ===")
    matching_times = []
    results = None

    for i in range(iterations):
        match_start = time.time()
        results = perform_matching(target_mfcc, source_mfcc, hop_length, sr)
        match_time = time.time() - match_start
        matching_times.append(match_time)
        print(f"第 {i+1}/{iterations} 次比对耗时: {match_time:.4f} 秒")

    # 计算统计信息
    avg_time = np.mean(matching_times)
    std_time = np.std(matching_times)
    min_time = np.min(matching_times)
    max_time = np.max(matching_times)
    total_time = np.sum(matching_times)

    print(f"\n=== 速度统计 ===")
    print(f"总比对次数: {iterations}")
    print(f"总比对时间: {total_time:.4f} 秒")
    print(f"平均比对时间: {avg_time:.4f} 秒")
    print(f"标准差: {std_time:.4f} 秒")
    print(f"最快比对时间: {min_time:.4f} 秒")
    print(f"最慢比对时间: {max_time:.4f} 秒")
    print(f"平均比对速度: {1/avg_time:.2f} 次/秒")

    # 显示最后一次比对的结果
    print("\n=== 比对结果 ===")
    print("\n整体匹配：")
    print(f"相似度 (余弦相似度): {results['overall_similarity']:.4f}")

    print("\n滑动窗口匹配：")
    print(f"查询片段长度: {results['query_frames']} 帧")
    print(f"目标音频长度: {results['target_frames']} 帧")
    print(f"\n最佳匹配位置: 帧 {results['best_position']} (时间: {results['best_time']:.2f}秒)")
    print(f"最佳匹配相似度: {results['best_similarity']:.4f}")
    print(f"匹配片段时间范围: {results['best_time']:.2f}秒 - {results['best_time'] + results['query_duration']:.2f}秒")

    # 总体时间统计
    total_elapsed = load_time + resample_time + mfcc_time + total_time
    print(f"\n=== 总体时间统计 ===")
    print(f"音频加载: {load_time:.4f} 秒 ({load_time/total_elapsed*100:.1f}%)")
    print(f"重采样: {resample_time:.4f} 秒 ({resample_time/total_elapsed*100:.1f}%)")
    print(f"MFCC提取: {mfcc_time:.4f} 秒 ({mfcc_time/total_elapsed*100:.1f}%)")
    print(f"比对计算: {total_time:.4f} 秒 ({total_time/total_elapsed*100:.1f}%)")
    print(f"总耗时: {total_elapsed:.4f} 秒")
    
    # CPU与GPU数据通信统计
    if data_transfer_stats['count'] > 0:
        print(f"\n=== CPU与GPU数据通信统计 ===")
        print(f"传输次数: {data_transfer_stats['count']}")
        print(f"传输总量: {data_transfer_stats['total_bytes'] / (1024**2):.2f} MB")
        print(f"传输总时间: {data_transfer_stats['total_time']:.4f} 秒")
        if data_transfer_stats['total_time'] > 0:
            bandwidth = (data_transfer_stats['total_bytes'] / (1024**2)) / data_transfer_stats['total_time']
            print(f"平均带宽: {bandwidth:.2f} MB/s")
        print(f"\n详细传输记录:")
        for i, transfer in enumerate(data_transfer_stats['transfers'], 1):
            print(f"  {i}. {transfer['name']}")
            print(f"     大小: {transfer['size_mb']:.2f} MB | 时间: {transfer['time']:.4f}秒 | {transfer['from']} -> {transfer['to']}")

    # 绘制相似度曲线和MFCC特征
    plt.figure(figsize=(14, 12))
    
    # 将MFCC数据转移到CPU用于绘图
    target_mfcc_cpu = target_mfcc.cpu().numpy()
    source_mfcc_cpu = source_mfcc.cpu().numpy()
    
    # 计算时间轴和统一的横坐标范围
    source_time_axis = np.arange(source_mfcc_cpu.shape[1]) * hop_length / sr
    target_time_axis = np.arange(target_mfcc_cpu.shape[1]) * hop_length / sr
    max_time = max(source_time_axis[-1], target_time_axis[-1])
    
    # 第一个子图：源音频MFCC
    plt.subplot(4, 1, 1)
    plt.imshow(source_mfcc_cpu, aspect='auto', origin='lower', cmap='viridis', 
               extent=[0, source_time_axis[-1], 0, n_mfcc])
    plt.colorbar(label='MFCC coefficient value')
    plt.xlim(0, max_time)  # 统一横坐标范围
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC coefficient')
    plt.title(f'Source MFCC (query clip) - duration: {source_time_axis[-1]:.2f}s')
    plt.grid(True, alpha=0.3)
    
    # 第二个子图：目标音频MFCC
    plt.subplot(4, 1, 2)
    plt.imshow(target_mfcc_cpu, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, target_time_axis[-1], 0, n_mfcc])
    plt.colorbar(label='MFCC coefficient value')
    # 标记最佳匹配区域
    match_end_time = results['best_time'] + results['query_duration']
    plt.axvline(x=results['best_time'], color='r', linestyle='--', linewidth=2,
                label=f'Match start ({results["best_time"]:.2f}s)')
    plt.axvline(x=match_end_time, color='orange', linestyle='--', linewidth=2,
                label=f'Match end ({match_end_time:.2f}s)')
    plt.axvspan(results['best_time'], match_end_time, alpha=0.2, color='red')
    plt.xlim(0, max_time)  # 统一横坐标范围
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC coefficient')
    plt.title(f'Target MFCC - duration: {target_time_axis[-1]:.2f}s')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 第三个子图：相似度曲线
    plt.subplot(4, 1, 3)
    time_positions = np.arange(len(results['similarities'])) * hop_length / sr
    plt.plot(time_positions, results['similarities'], linewidth=1)
    plt.axvline(x=results['best_time'], color='r', linestyle='--',
                label=f'Best match position ({results["best_time"]:.2f}s)')
    plt.axhline(y=results['best_similarity'], color='g', linestyle='--',
                alpha=0.5, label=f'Best similarity ({results["best_similarity"]:.4f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Similarity')
    plt.title('Sliding window similarity (PyTorch)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第四个子图：比对速度分布
    plt.subplot(4, 1, 4)
    plt.plot(range(1, iterations+1), matching_times, marker='o', markersize=4, linewidth=1)
    plt.axhline(y=avg_time, color='r', linestyle='--', alpha=0.7, 
                 label=f'Average time ({avg_time:.4f}s)')
    plt.fill_between(range(1, iterations+1), 
                     avg_time - std_time, avg_time + std_time, 
                     alpha=0.2, color='red', label='±1 standard deviation')
    plt.xlabel('Iteration')
    plt.ylabel('Duration (s)')
    plt.title(f'Timing across runs ({iterations} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mfcc_torch_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到 mfcc_torch_visualization.png")
