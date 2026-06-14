"""
=============================================================================
任务 A-1：数据清洗（豆瓣电影评分数据集）
=============================================================================
本脚本完成以下任务：
  1. 加载 CSV 数据到 PySpark DataFrame，打印 Schema 和前 5 行
  2. 统计各字段缺失值比例
  3. 对至少 2 个有缺失值的字段采用不同处理策略（dropna / fillna）
  4. 输出清洗前后行数对比及各字段基本统计信息

数据集字段说明：
  movie_id      - 电影唯一ID
  title         - 中文片名
  original_title- 原始片名
  year          - 上映年份
  rating_score  - 豆瓣评分
  rating_count  - 评分人数
  genres        - 类型（以 / 分隔）
  countries     - 制片国家/地区（以 / 分隔）
  directors     - 导演
  collect_count - 收藏人数
  summary       - 剧情简介
=============================================================================
"""

import sys, io, os
# Windows 控制台 UTF-8 编码修复
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', 'G:/anaconda/Library')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, mean, stddev, min, max
from pyspark.sql.types import FloatType, IntegerType, LongType

# ── 创建 Spark Session ──────────────────────────────────
spark = SparkSession.builder \
    .appName("DoubanMovies-DataCleaning") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# ── Step 1: 加载数据 ───────────────────────────────────
DATA_PATH = "C:/Users/admin/Desktop/cloud calculate/douban_movies.csv"
# 在 K8s 上运行时替换为：
# DATA_PATH = "s3a://<BUCKET>/douban_movies.csv"

df_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("encoding", "UTF-8") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("multiLine", "true") \
    .csv(DATA_PATH)

# 清理 BOM 字符（CSV 首列可能带 ﻿）
first_col = df_raw.columns[0]
if first_col.startswith('﻿'):
    new_name = first_col.lstrip('﻿')
    df_raw = df_raw.withColumnRenamed(first_col, new_name)

print("=" * 70)
print("A-1 Step 1: 加载数据完成")
print("=" * 70)
df_raw.printSchema()
print(f"\n总行数: {df_raw.count()}")
print("\n前5行数据:")
df_raw.show(5, truncate=50)

# ── Step 2: 缺失值分析 ─────────────────────────────────
print("\n" + "=" * 70)
print("A-1 Step 2: 缺失值统计")
print("=" * 70)

total_rows = df_raw.count()
missing_stats = []

for col_name in df_raw.columns:
    c = col(col_name)
    # 统一转字符串再比较，避免数字列和 "" 比较导致的 CAST_INVALID_INPUT
    missing_count = df_raw.filter(
        c.isNull() | (c.cast("string") == "") | (c.cast("string") == "null")
    ).count()
    missing_rate = missing_count / total_rows * 100
    missing_stats.append((col_name, missing_count, missing_rate))
    print(f"  {col_name:20s}: 缺失 {missing_count:6d} 行 ({missing_rate:5.2f}%)")

# ── Step 3: 数据清洗 ───────────────────────────────────
print("\n" + "=" * 70)
print("A-1 Step 3: 数据清洗")
print("=" * 70)

print(f"清洗前行数: {df_raw.count()}")

# 策略说明：
# 策略1 (dropna): 对 year 字段缺失的行直接删除
#   - 原因：年份是时间分析的关键字段，缺失会导致按年/月聚合结果偏差，且缺失量很少
# 策略2 (fillna): 对 summary 字段缺失的用 "暂无简介" 填充
#   - 原因：summary 是文本字段，不影响数值统计，填充默认值保留该行的其他有效信息

# 转换数值字段类型
df_clean = df_raw \
    .withColumn("year", col("year").cast(FloatType())) \
    .withColumn("rating_score", col("rating_score").cast(FloatType())) \
    .withColumn("rating_count", col("rating_count").cast(LongType())) \
    .withColumn("collect_count", col("collect_count").cast(LongType()))

# 策略1: 删除 year 为空的行
missing_year_before = df_clean.filter(col("year").isNull()).count()
df_clean = df_clean.filter(col("year").isNotNull())
print(f"  [策略1: dropna] 删除 year 为空的行: {missing_year_before} 行")

# 策略2: 填充 summary 为空的值
df_clean = df_clean.fillna({"summary": "暂无简介"})
print(f"  [策略2: fillna] 填充 summary 为空的值: \"暂无简介\"")

# 额外清洗：过滤评分异常值（0分通常是无效数据）
abnormal_scores = df_clean.filter(col("rating_score") < 1.0).count()
df_clean = df_clean.filter(col("rating_score") >= 1.0)
print(f"  [额外清洗] 删除 rating_score < 1.0 的异常行: {abnormal_scores} 行")

# 过滤 year < 1900 的数据（可能为录入错误）
abnormal_year = df_clean.filter(col("year") < 1900).count()
df_clean = df_clean.filter(col("year") >= 1900)
print(f"  [额外清洗] 删除 year < 1900 的异常行: {abnormal_year} 行")

print(f"\n清洗后行数: {df_clean.count()}")
print(f"共删除: {total_rows - df_clean.count()} 行")

# ── Step 4: 清洗后基本统计 ─────────────────────────────
print("\n" + "=" * 70)
print("A-1 Step 4: 清洗后基本统计信息")
print("=" * 70)

# 数值字段统计
numeric_cols = ["year", "rating_score", "rating_count", "collect_count"]
print("\n数值字段统计 (mean/stddev/min/max):")
for col_name in numeric_cols:
    stats = df_clean.select(
        mean(col(col_name)).alias("mean"),
        stddev(col(col_name)).alias("stddev"),
        min(col(col_name)).alias("min"),
        max(col(col_name)).alias("max")
    ).collect()[0]
    print(f"  {col_name:15s}: "
          f"mean={stats['mean']:.2f}, "
          f"stddev={stats['stddev']:.2f}, "
          f"min={stats['min']}, "
          f"max={stats['max']}")

# genres 字段分析（拆分多标签）
print("\n前10个最常见的电影类型:")
from pyspark.sql.functions import explode, split
genre_df = df_clean.select(explode(split(col("genres"), "/")).alias("genre")) \
    .groupBy("genre") \
    .count() \
    .orderBy(col("count").desc())
genre_df.show(10, truncate=False)

# 年份分布
print("\n年份分布（每10年统计）:")
from pyspark.sql.functions import floor
df_clean.withColumn("decade", floor(col("year") / 10) * 10) \
    .groupBy("decade") \
    .count() \
    .orderBy("decade") \
    .show(20, truncate=False)

print("\n数据清洗完成！")
spark.stop()
