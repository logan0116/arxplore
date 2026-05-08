"""
main.py — FastAPI 入口 + Lifespan 管理

启动：加载配置 → 初始化 SQLite → 连接 Qdrant → 初始化 embed_client → 启动调度器
关闭：清理资源
"""
import asyncio
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

import config, db, qdrant_store, embed_client, router, crawler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 启动阶段 ----
    cfg = config.load_config()

    # SQLite
    db_path = cfg["database"]["path"]
    conn = db.ensure_db(db_path)
    app.state.conn = conn

    # Qdrant
    qdrant = qdrant_store.QdrantStore(cfg)
    qdrant.init_collection()
    app.state.qdrant = qdrant

    # Embed client
    embed = embed_client.EmbedClient.from_config(cfg)
    app.state.embed = embed

    # Crawler
    c = crawler.Crawler(cfg, conn, qdrant)
    app.state.crawler = c

    # 调度器：每日凌晨 hour 点执行
    scheduler = AsyncIOScheduler()
    if cfg.get("scheduler", {}).get("enabled", True):
        hour = cfg.get("scheduler", {}).get("hour", 3)
        scheduler.add_job(c.run, "cron", hour=hour)

    app.state.scheduler = scheduler
    scheduler.start()

    yield

    # ---- 关闭阶段 ----
    scheduler.shutdown()
    await embed.close()
    conn.close()


app = FastAPI(lifespan=lifespan)

app.include_router(router.router, prefix="/arxplore")


if __name__ == "__main__":
    import uvicorn

    cfg = config.load_config()
    uvicorn.run(
        app=app,
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=False,
    )