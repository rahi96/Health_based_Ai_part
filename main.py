from fastapi import FastAPI
from ai.config import settings
from ai.routes import chat_routes
from ai.routes import cycle_awareness_routes
from ai.routes import daily_scripture_routes
from ai.routes import cycle_engine_v1_routes
from ai.routes import health_trends_routes
from ai.routes import numera_insight_routes
from ai.routes import summarize_pdf_routes
from ai.routes import skin_scan_routes
from ai.routes import smart_analysis_routes

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

app.include_router(
    cycle_engine_v1_routes.router,
    prefix="/api/v1/cycle-engine",
    tags=["Cycle_engine_v1"],
)
app.include_router(cycle_awareness_routes.router, prefix="/api", tags=["Cycle_awareness_api's"])
app.include_router(health_trends_routes.router, prefix="/api", tags=["Health_trends_api's"])
app.include_router(daily_scripture_routes.router, prefix="/api", tags=["Daily_scripture_api's"])
app.include_router(numera_insight_routes.router, prefix="/api", tags=["Numera_insight_api's"])
app.include_router(chat_routes.router, prefix="/api", tags=["Chatbot_api's"])
app.include_router(summarize_pdf_routes.router, prefix="/api", tags=["PDF_summary_api's"])
app.include_router(skin_scan_routes.router, prefix="/api", tags=["Skin_scan_api's"])
app.include_router(smart_analysis_routes.router, prefix="/api", tags=["Smart_analysis_api's"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
