import os
import redis
from flask import Flask, jsonify

app = Flask(__name__)

# Redis 配置（从环境变量读取，K8s 中由 ConfigMap/Secret 注入）
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

def get_redis_client():
    """创建 Redis 连接（有密码则认证）"""
    kwargs = {"host": REDIS_HOST, "port": REDIS_PORT, "decode_responses": True}
    if REDIS_PASSWORD:
        kwargs["password"] = REDIS_PASSWORD
    return redis.Redis(**kwargs)


@app.route("/api/ping", methods=["GET"])
def ping():
    """健康检查接口"""
    return jsonify({"status": "ok"})


@app.route("/api/counter", methods=["GET"])
def counter():
    """访问计数器（验证 Redis 连接）"""
    try:
        r = get_redis_client()
        count = r.incr("visit_count")
        return jsonify({"visit_count": count, "redis_host": REDIS_HOST})
    except redis.ConnectionError as e:
        return jsonify({"error": f"Redis connection failed: {str(e)}"}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """返回应用信息（验收用）"""
    return jsonify({
        "app": "cloud-course-backend",
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
