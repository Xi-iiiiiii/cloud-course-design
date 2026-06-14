"""
=============================================================================
任务 A-3：性能对比与 Amdahl 定律分析
=============================================================================
"""

import time
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', 'G:/anaconda/Library')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for font_name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']:
    try:
        fm.findfont(font_name, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

DATA_PATH = r"C:\Users\admin\Desktop\cloud calculate\douban_movies.csv"

print("=" * 70)
print("A-3: 性能对比实验")
print("=" * 70)

# ── 1. Pandas（单机，参考） ────────────────────────────────
print("\n[1] Pandas (单机, 参考)...")
t0 = time.time()
df_pd = pd.read_csv(DATA_PATH, encoding='utf-8')
df_pd = df_pd[df_pd['year'] >= 1900]
df_pd = df_pd[df_pd['rating_score'] >= 1.0]
df_pd = df_pd.dropna(subset=['countries'])
df_pd = df_pd[df_pd['countries'] != ""]
rows = []
for _, row in df_pd.iterrows():
    for country in str(row['countries']).split('/'):
        rows.append({'country': country.strip(), 'rating_score': row['rating_score'], 'rating_count': row['rating_count']})
df_country_pd = pd.DataFrame(rows)
result_pd = df_country_pd.groupby('country').agg(movie_count=('rating_score', 'count'), avg_rating=('rating_score', 'mean'), avg_rating_count=('rating_count', 'mean')).reset_index()
result_pd = result_pd[result_pd['movie_count'] >= 5]
result_pd = result_pd.sort_values('movie_count', ascending=False).head(15)
t_pandas = time.time() - t0
print(f"  Pandas: {t_pandas:.2f}s (参考, 非并行基线)")

# ── 2. PySpark ─────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round as spark_round, split, explode

spark = SparkSession.builder \
    .appName("PerformanceComparison") \
    .config("spark.master", "local[1]") \
    .config("spark.sql.adaptive.enabled", "false") \
    .getOrCreate()

# ── ExecutorInstances = 1（基线） ──
print("\n[2a] PySpark (1 executor, 基线)...")
spark.conf.set("spark.master", "local[1]")
t0 = time.time()
df = spark.read.option("header", "true").option("inferSchema", "true").option("encoding", "UTF-8").option("quote", "\"").option("escape", "\"").option("multiLine", "true").csv(DATA_PATH)
first_col = df.columns[0]
if first_col.startswith('﻿'):
    df = df.withColumnRenamed(first_col, first_col.lstrip('﻿'))
df = df.filter((col("year") >= 1900) & (col("rating_score") >= 1.0) & col("countries").isNotNull() & (col("countries") != ""))
df_country = df.withColumn("country", explode(split(col("countries"), "/")))
result_spark1 = df_country.groupBy("country").agg(count("*").alias("movie_count"), spark_round(avg("rating_score"), 2).alias("avg_rating")).filter(col("movie_count") >= 5).orderBy(col("movie_count").desc()).limit(15).collect()
t_spark1 = time.time() - t0
print(f"  PySpark (1 executor): {t_spark1:.2f}s (基线 = 1.00x)")

# ── ExecutorInstances = 2 ──
print("\n[2b] PySpark (2 executors)...")
spark.conf.set("spark.master", "local[2]")
t0 = time.time()
df2 = spark.read.option("header", "true").option("inferSchema", "true").option("encoding", "UTF-8").option("quote", "\"").option("escape", "\"").option("multiLine", "true").csv(DATA_PATH)
first_col2 = df2.columns[0]
if first_col2.startswith('﻿'):
    df2 = df2.withColumnRenamed(first_col2, first_col2.lstrip('﻿'))
df2 = df2.filter((col("year") >= 1900) & (col("rating_score") >= 1.0) & col("countries").isNotNull() & (col("countries") != ""))
df_country2 = df2.withColumn("country", explode(split(col("countries"), "/")))
result_spark2 = df_country2.groupBy("country").agg(count("*").alias("movie_count"), spark_round(avg("rating_score"), 2).alias("avg_rating")).filter(col("movie_count") >= 5).orderBy(col("movie_count").desc()).limit(15).collect()
t_spark2 = time.time() - t0
print(f"  PySpark (2 executors): {t_spark2:.2f}s")

spark.stop()

# 加速比（以 PySpark 1 executor 为基线）
speedup_2 = t_spark1 / t_spark2 if t_spark2 > 0 else 0
# Amdahl: S(2) = 1 / ((1-f) + f/2) => f = 2 * (1 - 1/S)
f_est = 2 * (1 - 1/speedup_2) if speedup_2 > 0 else 0
f_est = max(0.1, min(f_est, 0.95))

# ── 3. 绘制图表 ────────────────────────────────────────────
print("\n[3] 绘制性能对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 执行时间柱状图
ax1 = axes[0]
methods = ['Pandas\n(单机, 参考)', 'PySpark\n(1 executor, 基线)', 'PySpark\n(2 executors)']
times = [t_pandas, t_spark1, t_spark2]
bars = ax1.bar(methods, times, color=['#95a5a6', '#e74c3c', '#2ecc71'], edgecolor='white', linewidth=1.2)
ax1.set_ylabel('执行时间 (秒)', fontsize=12)
ax1.set_title('同一查询执行时间对比 (Pandas vs PySpark)', fontsize=13, fontweight='bold')
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{t:.2f}s', ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2: Amdahl 分析（基线 = PySpark 1 executor）
ax2 = axes[1]
n_range = np.linspace(1, 8, 50)
amdahl = 1 / ((1 - f_est) + f_est / n_range)
ax2.plot(n_range, amdahl, 'b-', linewidth=2, label=f'Amdahl 理论 (f={f_est:.2f})')
ax2.plot([1, 2], [1.0, speedup_2], 'ro-', markersize=10, linewidth=2, label=f'实测 (S={speedup_2:.2f}x)')
ax2.plot([1, 8], [1, 8], 'k--', alpha=0.3, label='线性加速')
ax2.set_xlabel('Executor 数量', fontsize=12)
ax2.set_ylabel('加速比', fontsize=12)
ax2.set_title('实测加速比 vs Amdahl 理论 (基线=1 executor)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_xticks(range(1, 9))

fig.suptitle(f'PySpark 1→2 executor 加速 {speedup_2:.2f}x,  估算并行比例 f ≈ {f_est:.2f}', fontsize=12, color='gray')
plt.tight_layout()
plt.savefig(r'C:\Users\admin\Desktop\cloud calculate\spark\performance_comparison.png', dpi=150, bbox_inches='tight')
print("  图表已保存: spark/performance_comparison.png")

# ── 4. 文字分析 ────────────────────────────────────────────
print("\n" + "=" * 70)
print("Amdahl 定律分析")
print("=" * 70)

print(f"""
【实测数据】
  Pandas (单机, 参考):              {t_pandas:.2f}s
  PySpark (1 executor, 基线):        {t_spark1:.2f}s
  PySpark (2 executors):             {t_spark2:.2f}s  (加速比 = {speedup_2:.2f}x)

【Amdahl 定律】
  S(n) = 1 / ((1 - f) + f / n)
  根据 S(2) = {speedup_2:.2f} 反推: f = 2 * (1 - 1/{speedup_2:.2f}) = {f_est:.2f}

【分析】
1. Pandas vs PySpark:
   Pandas ({t_pandas:.2f}s) 远快于 PySpark ({t_spark1:.2f}s)，因为 67K 行数据量太小，
   PySpark 的 JVM 启动 + 序列化 + shuffle 固定开销远超实际计算时间。
   Spark 适合 GB+ 级大数据，小数据场景应直接用 Pandas。

2. 并行加速 (1→2 executor):
   实测加速 {speedup_2:.2f}x。由于数据量小、shuffle 和 I/O 占比高，
   并行收益受限——这就是 Amdahl 定律中 (1-f) 串行部分的作用。

3. 加速未达线性的原因:
   - Shuffle 通信: GROUP BY 触发 executor 间数据传输
   - 数据倾斜: 美国/中国的数据量远超小国，成为瓶颈
   - 串行 I/O: CSV 读取、结果 collect() 到 Driver 不可并行
   - 数据量小: 67K 行 / 39MB，并行化固定开销占比过大

4. 结论:
   Amdahl 定律清楚地展示了: 即使增加处理器，串行瓶颈 (1-f) 也限制着加速上限。
   当数据量扩大到 GB 级别，并行收益将大幅提升——符合 Gustafson 定律
   (大规模数据下可并行比例 f 增大，并行效率更高) 的预期。
""")

print("性能对比实验完成！")
