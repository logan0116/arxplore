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
from logging_config import setup_logging, get_logger
from errors import ErrorCode, ArxploreError

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("ArXplore 服务启动中...")
    logger.info("=" * 60)

    # ---- 启动阶段 ----
    cfg = config.load_config()

    # SQLite
    db_path = cfg["database"]["path"]
    logger.info(f"初始化 SQLite: {db_path}")
    conn = db.ensure_db(db_path)
    app.state.conn = conn
    logger.info("SQLite 初始化完成")

    # Qdrant
    logger.info(f"连接 Qdrant: {cfg['qdrant']['host']}:{cfg['qdrant']['port']}")
    qdrant = qdrant_store.QdrantStore(cfg)
    qdrant.init_collection()
    app.state.qdrant = qdrant
    logger.info("Qdrant 初始化完成")

    # Embed client
    logger.info(f"连接 Embedding 服务: {cfg['embedding']['base_url']}")
    embed = embed_client.EmbedClient.from_config(cfg)
    app.state.embed = embed
    logger.info("Embedding 客户端初始化完成")

    # Crawler
    c = crawler.Crawler(cfg, conn, qdrant)
    app.state.crawler = c
    logger.info("爬虫初始化完成")

    # 调度器
    scheduler = AsyncIOScheduler()
    if cfg.get("scheduler", {}).get("enabled", True):
        hour = cfg.get("scheduler", {}).get("hour", 3)
        scheduler.add_job(c.run, "cron", hour=hour)
        logger.info(f"调度器已启动，每日 {hour}:00 执行采集")

    app.state.scheduler = scheduler
    scheduler.start()

    logger.info("=" * 60)
    logger.info("ArXplore 服务启动完成")
    logger.info("=" * 60)

    yield

    # ---- 关闭阶段 ----
    logger.info("ArXplore 服务关闭中...")
    scheduler.shutdown()
    logger.info("调度器已关闭")
    await embed.close()
    logger.info("Embedding 客户端已关闭")
    conn.close()
    logger.info("SQLite 连接已关闭")
    logger.info("ArXplore 服务已关闭")


app = FastAPI(lifespan=lifespan)

app.include_router(router.router, prefix="/arxplore")


@app.exception_handler(ArxploreError)
async def arxplore_error_handler(request, exc: ArxploreError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.code.value,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


if __name__ == "__main__":
    import uvicorn

    cfg = config.load_config()
    uvicorn.run(
        app=app,
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=False,
    )
