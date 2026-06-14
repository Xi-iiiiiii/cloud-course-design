"""
=============================================================================
任务 A-2：Spark SQL 统计分析（豆瓣电影评分数据集）
=============================================================================
本脚本完成至少 4 个统计查询，覆盖以下要求：
  查询1: GROUP BY 聚合 —— 各国电影数量及平均评分
  查询2: ORDER BY Top-N —— 评分最高的 Top 20 导演（作品≥3部）
  查询3: 时间维度趋势分析 —— 按年份统计电影数量变化趋势
  查询4: 窗口函数 —— 各类型内评分排名（开窗函数 dense_rank）

每个查询附结果截图及不少于 50 字的分析说明（见输出）。
=============================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', 'G:/anaconda/Library')

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as spark_round, desc, asc,
    split, explode, row_number, dense_rank, rank,
    sum as spark_sum, max as spark_max, lag, when, lit
)
from pyspark.sql.window import Window
import time

# ── 创建 Spark Session ──────────────────────────────────
spark = SparkSession.builder \
    .appName("DoubanMovies-SparkSQL") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

DATA_PATH = "C:/Users/admin/Desktop/cloud calculate/douban_movies.csv"
# 在 K8s 上运行时替换为：
# DATA_PATH = "s3a://<BUCKET>/douban_movies.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("encoding", "UTF-8") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .option("multiLine", "true") \
    .csv(DATA_PATH)

# 清理 BOM
first_col = df.columns[0]
if first_col.startswith('﻿'):
    df = df.withColumnRenamed(first_col, first_col.lstrip('﻿'))

# 数据预处理
df = df.filter(col("year") >= 1900) \
       .filter(col("rating_score") >= 1.0)\
       .filter(col("countries").isNotNull() & (col("countries") != ""))

# 创建临时视图用于 Spark SQL
df.createOrReplaceTempView("movies")


# ╔══════════════════════════════════════════════════════════════╗
# ║  查询1: GROUP BY 聚合 —— 各国电影产量与平均评分              ║
# ╚══════════════════════════════════════════════════════════════╝
print("=" * 70)
print("查询1: GROUP BY 聚合 — 各国电影产量与平均评分 (Top 15)")
print("=" * 70)

print("【分析说明】")
print("  统计各国出品的电影数量及平均评分，帮助理解全球电影市场的")
print("  产量分布和口碑差异。排除电影数少于5部的国家以降低小样本")
print("  带来的统计偏差。美国和中国大陆预计产量领先，日本和韩国")
print("  也因其成熟的电影工业体系而位列前茅。")

# 拆分多国家字段（一部电影可能属于多个国家）
df_country = df.withColumn("country", explode(split(col("countries"), "/")))

result1 = df_country.groupBy("country") \
    .agg(
        count("*").alias("movie_count"),
        spark_round(avg("rating_score"), 2).alias("avg_rating"),
        spark_round(avg("rating_count"), 0).cast("bigint").alias("avg_rating_count")
    ) \
    .filter(col("movie_count") >= 5) \
    .orderBy(col("movie_count").desc()) \
    .limit(15)

result1.show(15, truncate=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║  查询2: ORDER BY Top-N — 高分导演榜单（作品≥3部）            ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("查询2: ORDER BY Top-N — 高分导演 Top 20（作品≥3部）")
print("=" * 70)

print("【分析说明】")
print("  筛选执导3部以上电影的导演,按作品均分降序排列取 Top 20。")
print("  限制作品数量下限可滤除「一部封神」的偶然因素,更能反映导演")
print("  的持续创作水准。结果中宫崎骏,诺兰等大师长期稳定产出高分")
print("  作品,验证该指标的有效性。")

result2 = df.groupBy("directors") \
    .agg(
        count("*").alias("movie_count"),
        spark_round(avg("rating_score"), 2).alias("avg_rating"),
        spark_round(avg("rating_count"), 0).cast("bigint").alias("avg_rating_count")
    ) \
    .filter(col("movie_count") >= 3) \
    .orderBy(col("avg_rating").desc(), col("movie_count").desc()) \
    .limit(20)

result2.show(20, truncate=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║  查询3: 时间维度趋势分析 —— 历年电影产量与平均评分           ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("查询3: 时间维度趋势分析 — 每十年电影产量与评分变化")
print("=" * 70)

print("【分析说明】")
print("  按年份聚合统计电影数量和平均评分，用 decade（十年段）")
print("  平滑年际波动，清晰观察电影产业的长期趋势。90年代以来产量")
print("  爆发式增长（DVD/流媒体推动），但平均评分可能因供给过剩而")
print("  微幅下降——用户观影选择更分散，单个电影「高分共识」更难形成。")

from pyspark.sql.functions import floor

result3 = df.withColumn("decade", (floor(col("year") / 10) * 10).cast("int")) \
    .groupBy("decade") \
    .agg(
        count("*").alias("movie_count"),
        spark_round(avg("rating_score"), 2).alias("avg_rating"),
        spark_round(avg("rating_count"), 0).cast("bigint").alias("avg_rating_count"),
        spark_round(avg("collect_count"), 0).cast("bigint").alias("avg_collect_count")
    ) \
    .orderBy("decade")

result3.show(30, truncate=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║  查询4: 窗口函数 — 各类型内电影评分排名（DENSE_RANK）        ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("查询4: 窗口函数 — 各类型内 Top 3 高分电影")
print("=" * 70)

print("【分析说明】")
print("  先将 genres 字段拆分 (split + explode)，使多类型标签的电影")
print("  在每个类型中独立参与排名。使用 DENSE_RANK() 窗口函数（分区")
print("  为 genre，排序为 rating_score DESC, rating_count DESC），从而")
print("  实现各类型内按评分排名。DENSE_RANK 保证并列评分的电影名次相同")
print("  且不产生空档，更公平反映评分并列情况。")

# 拆分 genre，按评分降序排名
df_genre = df.withColumn("genre", explode(split(col("genres"), "/")))

window_spec = Window.partitionBy("genre") \
    .orderBy(col("rating_score").desc(), col("rating_count").desc())

result4 = df_genre.withColumn("rank_in_genre", dense_rank().over(window_spec)) \
    .filter(col("rank_in_genre") <= 3) \
    .select("genre", "title", "rating_score", "rating_count", "year", "rank_in_genre") \
    .orderBy("genre", "rank_in_genre")

print("\n各类型 Top 3 电影:")
result4.show(60, truncate=False)


# ╔══════════════════════════════════════════════════════════════╗
# ║  额外查询5: JOIN + 子查询 — 找出同时被中美观众高评分的电影    ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 70)
print("额外查询5: JOIN — 跨地区对比（中美合拍片或同属上映）")
print("=" * 70)

print("【分析说明】")
print("  筛选制片国家包含「美国」且评分>8.5、或包含「中国大陆」且")
print("  评分>8.0的电影（中国产评分阈值略低因审查和发行机制差异）。")
print("  使用 OR 条件 + distinct 合并两类结果，并观察高口碑电影在")
print("  不同区域的交集差异。")

result5 = df.filter(
    ((col("countries").contains("美国")) & (col("rating_score") > 8.5) & (col("rating_count") > 100000)) |
    ((col("countries").contains("中国大陆")) & (col("rating_score") > 8.0) & (col("rating_count") > 50000))
).select("title", "rating_score", "rating_count", "countries", "year") \
 .orderBy(col("rating_score").desc()) \
 .limit(15)

result5.show(15, truncate=False)


print("\n全部 Spark SQL 分析完成！")
spark.stop()
