import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 TorchAudio 版本
print("TorchAudio 版本:", torchaudio.__version__)

# 检查 CUDA 是否可用（可选）
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)

# 简单测试：创建一个音频张量（模拟 1 秒 16kHz 单声道音频）
sample_rate = 16000
duration = 1  # 秒
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440Hz 正弦波，单声道

print("生成的音频张量形状:", waveform.shape)
print("采样率:", sample_rate)

# 可选：保存为 WAV 文件（需要 torchaudio 支持）
try:
    torchaudio.save("test_tone.wav", waveform, sample_rate)
    print("音频已保存为 test_tone.wav")
except Exception as e:
    print("保存音频时出错:", e)

# ===== 计算并显示 MFCC =====
print("\n计算 MFCC 特征...")

# MFCC 参数
n_mfcc = 40
n_fft = 512  # 25ms
hop_length = int(sample_rate * 0.01)  # 10ms
win_length = int(sample_rate * 0.025)  # 25ms

# 创建 MFCC 转换器
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length
    }
)

# 计算 MFCC
mfcc = mfcc_transform(waveform)  # (channel, n_mfcc, time)
print(f"MFCC 形状: {mfcc.shape}")

# 转换为 numpy 用于绘图
mfcc_np = mfcc.squeeze(0).numpy()  # (n_mfcc, time)

# 创建可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. 绘制原始波形
axes[0].plot(t.numpy(), waveform.squeeze(0).numpy())
axes[0].set_title('原始音频波形 (440Hz 正弦波)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('时间 (秒)')
axes[0].set_ylabel('振幅')
axes[0].grid(True, alpha=0.3)

# 2. 绘制 MFCC 热图
time_frames = np.arange(mfcc_np.shape[1]) * hop_length / sample_rate
im = axes[1].imshow(
    mfcc_np,
    aspect='auto',
    origin='lower',
    extent=[0, duration, 0, n_mfcc],
    cmap='viridis'
)
axes[1].set_title('MFCC 特征热图', fontsize=14, fontweight='bold')
axes[1].set_xlabel('时间 (秒)')
axes[1].set_ylabel('MFCC 系数')
plt.colorbar(im, ax=axes[1], label='幅度')

# 3. 绘制前几个 MFCC 系数随时间的变化
n_coeff_to_plot = 13
for i in range(n_coeff_to_plot):
    axes[2].plot(time_frames, mfcc_np[i, :], label=f'MFCC {i+1}', alpha=0.7)
axes[2].set_title(f'前 {n_coeff_to_plot} 个 MFCC 系数随时间变化', fontsize=14, fontweight='bold')
axes[2].set_xlabel('时间 (秒)')
axes[2].set_ylabel('MFCC 值')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mfcc_visualization.png', dpi=150, bbox_inches='tight')
print("\nMFCC 可视化已保存为 mfcc_visualization.png")


# 打印统计信息
print("\nMFCC 统计信息:")
print(f"  平均值: {mfcc_np.mean():.4f}")
print(f"  标准差: {mfcc_np.std():.4f}")
print(f"  最小值: {mfcc_np.min():.4f}")
print(f"  最大值: {mfcc_np.max():.4f}")
print(f"  时间帧数: {mfcc_np.shape[1]}")
