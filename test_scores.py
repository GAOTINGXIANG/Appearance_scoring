import numpy as np

print("=== 评分曲线验证 ===\n")

# 1. 对称性评分曲线
print("1. 对称性评分 (base=0.4, scale=22)")
print("   误差(px) -> 得分(25分制)")
sym_base = 0.4
sym_scale = 22.0
for err in [0, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 165]:
    score_raw = sym_base + (1.0 - sym_base) * np.exp(-err / sym_scale)
    score_raw = max(0.35, min(1.0, score_raw))
    score_25 = score_raw * 25
    print(f"   {err:3d}px -> raw={score_raw:.4f} -> {score_25:.1f}/25")

print()

# 2. 嘴唇饱满度评分曲线
print("2. 嘴唇饱满度 (ideal=0.22, sigma=0.18)")
print("   lip_ratio -> 得分(10分制)")
ideal_lip = 0.22
sigma_lip = 0.18
for ratio in [0.0, 0.05, 0.10, 0.15, 0.20, 0.22, 0.30, 0.40, 0.50]:
    lip_raw = np.exp(-0.5 * ((ratio - ideal_lip) / sigma_lip) ** 2)
    lip_score = max(0.45, lip_raw)
    print(f"   {ratio:.2f} -> raw={lip_raw:.4f} -> {lip_score*10:.1f}/10")

print()

# 3. 眉眼协调评分
print("3. 眉眼协调 (ratio_ideal=1.15, sigma=0.55, diff_scale=40, diff_base=0.40)")
print("   brow_eye_ratio, brow_height_diff -> 得分(5分制)")
ideal_ratio = 1.15
sigma_ratio = 0.55
diff_scale = 40.0
diff_base = 0.40
for ratio in [0.5, 1.0, 1.15, 1.5, 2.0, 2.5, 3.0]:
    for diff in [0, 10, 20, 30, 50, 80]:
        ratio_raw = np.exp(-0.5 * ((ratio - ideal_ratio) / sigma_ratio) ** 2)
        ratio_score = max(0.45, min(1.0, ratio_raw))
        diff_score = diff_base + (1.0 - diff_base) * np.exp(-diff / diff_scale)
        diff_score = max(0.40, min(1.0, diff_score))
        brow_score = 0.7 * ratio_score + 0.3 * diff_score
        brow_score = max(0.45, min(1.0, brow_score))
        print(f"   ratio={ratio:.2f}, diff={diff:2d}px -> {brow_score*5:.1f}/5")
    print()

# 4. 验证截图中的数值
print("=== 截图数值反推 ===")
print("对称误差 165.41px:")
err = 165.41
score_raw = sym_base + (1.0 - sym_base) * np.exp(-err / sym_scale)
score_raw = max(0.35, min(1.0, score_raw))
print(f"   计算结果: raw={score_raw:.4f} -> {score_raw*25:.1f}/25")
print(f"   截图显示: 0.4326 -> {0.4326*25:.1f}/25")
print()
print("嘴唇饱满度 0.0:")
ratio = 0.0
lip_raw = np.exp(-0.5 * ((ratio - ideal_lip) / sigma_lip) ** 2)
lip_score = max(0.45, lip_raw)
print(f"   计算结果: raw={lip_raw:.4f} -> {lip_score*10:.1f}/10")
print(f"   截图显示: 0.0 -> 0.0/10")
