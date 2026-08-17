"""
app.py
Minimal FastAPI application exposing the two trained models as REST
endpoints. This is what the Dockerfile runs in production.

Endpoints
---------
GET  /health                 -> liveness check
POST /predict/segment        -> classify customer segment
POST /predict/lead-time      -> predict delivery lead time in days
"""

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_lead_time, predict_segment

app = FastAPI(
    title="Delivery Intelligence ML API",
    description="Predicts customer segment and delivery lead time from order attributes.",
    version="1.0.0",
)


class SegmentRequest(BaseModel):
    quantity: float = Field(..., ge=0, example=250)
    delivery_lead_time: float = Field(..., ge=0, example=12)
    price: float = Field(..., ge=0, example=62.3)
    category: str = Field(..., example="Noodles")
    carrier: str = Field(..., example="akr express")
    city_tier: str = Field(..., example="Tier-2")


class LeadTimeRequest(BaseModel):
    quantity: float = Field(..., ge=0, example=250)
    price: float = Field(..., ge=0, example=62.3)
    category: str = Field(..., example="Noodles")
    carrier: str = Field(..., example="akr express")
    city_tier: str = Field(..., example="Tier-2")
    segment: str = Field(..., example="Loyal")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/segment")
def segment_endpoint(payload: SegmentRequest):
    try:
        prediction = predict_segment(payload.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"predicted_segment": prediction[0]}


@app.post("/predict/lead-time")
def lead_time_endpoint(payload: LeadTimeRequest):
    try:
        prediction = predict_lead_time(payload.dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"predicted_delivery_lead_time_days": round(prediction[0], 2)}
