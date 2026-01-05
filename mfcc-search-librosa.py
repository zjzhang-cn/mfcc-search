import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import time

# 设置中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 黑体或微软雅黑
#plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 解析命令行参数
parser = argparse.ArgumentParser(description='音频MFCC特征提取和相似度计算')
parser.add_argument('-t', '--target', required=True, help='目标音频文件路径')
parser.add_argument('-s', '--source', required=True, help='查询音频文件路径')
parser.add_argument('-n', '--iterations', type=int, default=10, help='比对次数 (默认: 10)')
args = parser.parse_args()

print(f"=== 开始加载音频 ===")
load_start = time.time()

sr = 16000
n_mfcc = 40
target_file = args.target
source_file = args.source
n_fft = 512  # 25ms
hop_length = int(sr*0.01)  # 10ms
win_length = int(sr*0.025)  # 25ms
iterations = args.iterations

y, sr = librosa.load(target_file, sr=sr)
target_mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=n_mfcc,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length
)

# 显示MFCC热图
# plt.figure(figsize=(12, 4))
# librosa.display.specshow(
#     target_mfcc,
#     x_axis='time',  # X轴显示时间
#     sr=sr,
#     hop_length=512
# )
# plt.colorbar()
# plt.title(f'MFCC 时间序列 - {target_file}')
# plt.ylabel('MFCC系数')
# plt.xlabel('时间 (秒)')
# plt.tight_layout()


y, sr = librosa.load(source_file, sr=sr)
source_mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=n_mfcc,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length
)

load_time = time.time() - load_start
print(f"音频加载和MFCC提取时间: {load_time:.4f} 秒")


def perform_matching(target_mfcc, source_mfcc, hop_length, sr):
    """执行一次完整的匹配过程并返回结果"""
    # 整体匹配：直接展平后计算余弦相似度
    min_time = min(target_mfcc.shape[1], source_mfcc.shape[1])
    target_flat = target_mfcc[:, :min_time].flatten()
    source_flat = source_mfcc[:, :min_time].flatten()
    
    cosine_distance = cosine(target_flat, source_flat)
    overall_similarity = 1 - cosine_distance

    # 滑动窗口匹配
    query_frames = source_mfcc.shape[1]
    target_frames = target_mfcc.shape[1]

    if query_frames > target_frames:
        target_mfcc, source_mfcc = source_mfcc, target_mfcc
        query_frames = source_mfcc.shape[1]
        target_frames = target_mfcc.shape[1]

    # 对查询片段进行归一化和统计特征提取
    query_normalized = (source_mfcc - np.mean(source_mfcc, axis=1, keepdims=True)
                        ) / (np.std(source_mfcc, axis=1, keepdims=True) + 1e-8)
    query_flat = query_normalized.flatten()

    # 提取查询片段的统计特征
    query_stats = np.concatenate([
        np.mean(source_mfcc, axis=1),
        np.std(source_mfcc, axis=1),
        np.max(source_mfcc, axis=1),
        np.min(source_mfcc, axis=1),
    ])

    best_similarity = -1.0
    best_position = 0
    similarities = []

    # 滑动窗口遍历
    for i in range(target_frames - query_frames + 1):
        window = target_mfcc[:, i:i+query_frames]

        window_normalized = (window - np.mean(window, axis=1, keepdims=True)
                             ) / (np.std(window, axis=1, keepdims=True) + 1e-8)
        window_flat = window_normalized.flatten()

        window_stats = np.concatenate([
            np.mean(window, axis=1),
            np.std(window, axis=1),
            np.max(window, axis=1),
            np.min(window, axis=1),
        ])

        cos_dist_norm = cosine(window_flat, query_flat)
        sim_norm = 1 - cos_dist_norm

        cos_dist_stats = cosine(window_stats, query_stats)
        sim_stats = 1 - cos_dist_stats

        sim = 0.7 * sim_norm + 0.3 * sim_stats
        similarities.append(sim)

        if sim > best_similarity:
            best_similarity = sim
            best_position = i

    best_time = best_position * hop_length / sr
    query_duration = query_frames * hop_length / sr

    return {
        'overall_similarity': overall_similarity,
        'cosine_distance': cosine_distance,
        'best_similarity': best_similarity,
        'best_position': best_position,
        'best_time': best_time,
        'query_duration': query_duration,
        'query_frames': query_frames,
        'target_frames': target_frames,
        'similarities': similarities
    }


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
print(f"余弦距离: {results['cosine_distance']:.4f}")
print(f"相似度 (余弦相似度): {results['overall_similarity']:.4f}")

print("\n滑动窗口匹配：")
print(f"查询片段长度: {results['query_frames']} 帧")
print(f"目标音频长度: {results['target_frames']} 帧")
print(f"\n最佳匹配位置: 帧 {results['best_position']} (时间: {results['best_time']:.2f}秒)")
print(f"最佳匹配相似度: {results['best_similarity']:.4f}")
print(f"匹配片段时间范围: {results['best_time']:.2f}秒 - {results['best_time'] + results['query_duration']:.2f}秒")

# 总体时间统计
total_elapsed = load_time + total_time
print(f"\n=== 总体时间统计 ===")
print(f"音频加载和MFCC提取: {load_time:.4f} 秒 ({load_time/total_elapsed*100:.1f}%)")
print(f"比对计算: {total_time:.4f} 秒 ({total_time/total_elapsed*100:.1f}%)")
print(f"总耗时: {total_elapsed:.4f} 秒")

# 绘制相似度曲线和速度分析
plt.figure(figsize=(12, 8))

# 第一个子图：相似度曲线
plt.subplot(2, 1, 1)
time_positions = np.arange(len(results['similarities'])) * hop_length / sr
plt.plot(time_positions, results['similarities'], linewidth=1)
plt.axvline(x=results['best_time'], color='r', linestyle='--',
          label=f'Best match position ({results["best_time"]:.2f}s)')
plt.axhline(y=results['best_similarity'], color='g', linestyle='--',
          alpha=0.5, label=f'Best similarity ({results["best_similarity"]:.4f})')
plt.xlabel('Time (s)')
plt.ylabel('Similarity')
plt.title('Sliding window similarity (Librosa)')
plt.legend()
plt.grid(True, alpha=0.3)

# 第二个子图：比对速度分布
plt.subplot(2, 1, 2)
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
plt.savefig('mfcc_librosa_visualization.png', dpi=150, bbox_inches='tight')
print(f"\n结果已保存到 mfcc_librosa_visualization.png")