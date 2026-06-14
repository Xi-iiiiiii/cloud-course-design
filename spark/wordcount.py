"""
=============================================================================
任务 A-0：WordCount 验证作业（Spark Operator 环境验证用）
=============================================================================
提交到 Spark Operator 验证集群环境是否就绪。
Driver Pod 状态变为 Completed 即表示环境正常。
=============================================================================
"""

import os
os.environ['JAVA_HOME'] = os.environ.get('JAVA_HOME', 'G:/anaconda/Library')

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("WordCount") \
    .config("spark.executor.memory", "512m") \
    .getOrCreate()

# 读取示例文本（OBS 路径由教师提供）
# 本地测试时使用文件系统中的文本文件
import os
import sys

# 优先使用 OBS 路径，不存在则使用本地文件
TEXT_PATH = os.environ.get("TEXT_PATH", "s3a://<BUCKET>/sample.txt")

# 如果 s3a 路径不可用，生成内置文本进行测试
try:
    lines = spark.sparkContext.textFile(TEXT_PATH)
except Exception:
    print(f"无法访问 {TEXT_PATH}，使用内置测试文本")
    test_text = ["hello spark", "hello world", "spark is fast",
                 "hello k8s", "spark on k8s", "world hello"]
    lines = spark.sparkContext.parallelize(test_text)

word_counts = (
    lines.flatMap(lambda line: line.split())
         .map(lambda word: (word, 1))
         .reduceByKey(lambda a, b: a + b)
         .sortBy(lambda x: x[1], ascending=False)
)

print("Top 10 words:", word_counts.take(10))
spark.stop()
