# File: demo_target/main.py
# Description: Implements a sample order API used for demonstrations and tests.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

"""Purpose-built demo target (synthetic data only).

Behaviour: auth (401), ownership (404), stock (409), declined card (402), quantity limit (422),
idempotency keys, cancellation with stock restore, server-side pricing and a small shop UI.
Seed defects with DEMO_DEFECTS (comma list): quantity_limit, price_tamper, idempotency, server_error, ui_confirmation.
"""
from __future__ import annotations

import os
from copy import deepcopy
from threading import Lock
from typing import Dict, Optional, Set, Tuple
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

Body = Dict[str, object]
Store = Dict[str, Body]
IdempotencyIndex = Dict[Tuple[str, str], str]

TOKENS = {"demo-customer-token": "cust-1", "demo-other-token": "cust-2"}
DECLINED_CARD = "4000000000000002"
MAX_QUANTITY = 5
INITIAL_PRODUCTS: Store = {
    "toy-001": {"id": "toy-001", "name": "Wooden train", "price": 19.99, "stock": 100},
    "toy-002": {"id": "toy-002", "name": "Puzzle cube", "price": 9.5, "stock": 0},
    "toy-003": {"id": "toy-003", "name": "Kite", "price": 12.0, "stock": 1},
}


class OrderRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=80)
    quantity: int = Field(ge=1)
    payment_card: str = Field(default="4242424242424242", min_length=12, max_length=19)
    unit_price: Optional[float] = None  # ignored unless the price_tamper defect is seeded


def _customer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    customer = TOKENS.get(authorization.removeprefix("Bearer ").strip())
    if customer is None:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return customer


def _public(order: Body) -> Body:
    return {key: value for key, value in order.items() if key != "customer"}


def create_demo_app(defects: Optional[Set[str]] = None) -> FastAPI:
    active = set(defects) if defects is not None else {
        d.strip() for d in os.getenv("DEMO_DEFECTS", "").split(",") if d.strip()}
    app = FastAPI(title="AI QA Platform Demo Target", version="2.0.0")
    lock = Lock()
    products = deepcopy(INITIAL_PRODUCTS)
    orders: Store = {}
    idempotency: IdempotencyIndex = {}

    def owned(order_id: str, customer: str) -> Body:
        order = orders.get(order_id)
        if order is None or order["customer"] != customer:
            raise HTTPException(status_code=404, detail="Order not found.")  # never reveal existence
        return order

    @app.get("/products/{product_id}")
    def get_product(product_id: str) -> Body:
        with lock:
            product = products.get(product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found.")
            return dict(product)

    @app.post("/orders", status_code=status.HTTP_201_CREATED)
    def create_order(payload: OrderRequest, authorization: Optional[str] = Header(default=None),
                     idempotency_key: Optional[str] = Header(default=None)) -> Body:
        customer = _customer(authorization)
        if payload.quantity > MAX_QUANTITY and "quantity_limit" not in active:
            raise HTTPException(status_code=422, detail=f"Quantity must be at most {MAX_QUANTITY}.")
        with lock:
            if idempotency_key and "idempotency" not in active:
                existing = idempotency.get((customer, idempotency_key))
                if existing:
                    return _public(orders[existing])
            product = products.get(payload.product_id)
            if product is None:
                raise HTTPException(status_code=404, detail="Product not found.")
            stock = int(str(product["stock"]))
            if stock < payload.quantity:
                raise HTTPException(status_code=409, detail="Out of stock.")
            if payload.payment_card.replace(" ", "") == DECLINED_CARD:
                raise HTTPException(status_code=402, detail="Payment declined.")
            unit_price = float(str(product["price"]))
            if "price_tamper" in active and payload.unit_price is not None:
                unit_price = payload.unit_price
            product["stock"] = stock - payload.quantity
            order_id = str(uuid4())
            order: Body = {
                "id": order_id, "customer": customer, "product_id": payload.product_id,
                "quantity": payload.quantity, "unit_price": unit_price,
                "total": round(unit_price * payload.quantity, 2), "status": "confirmed",
            }
            orders[order_id] = order
            if idempotency_key:
                idempotency[(customer, idempotency_key)] = order_id
            return _public(order)

    @app.get("/orders/{order_id}")
    def get_order(order_id: str, authorization: Optional[str] = Header(default=None)) -> Body:
        customer = _customer(authorization)
        if "server_error" in active:
            raise HTTPException(status_code=500, detail="Internal error.")
        with lock:
            return _public(owned(order_id, customer))

    @app.post("/orders/{order_id}/cancel")
    def cancel_order(order_id: str, authorization: Optional[str] = Header(default=None)) -> Body:
        customer = _customer(authorization)
        with lock:
            order = owned(order_id, customer)
            if order["status"] != "confirmed":
                raise HTTPException(status_code=409, detail="Only confirmed orders can be cancelled.")
            order["status"] = "cancelled"
            product = products[str(order["product_id"])]
            product["stock"] = int(str(product["stock"])) + int(str(order["quantity"]))
            return _public(order)

    @app.get("/shop", response_class=HTMLResponse)
    def shop() -> str:
        word = "pending" if "ui_confirmation" in active else "confirmed"
        options = "".join(f'<option value="{p["id"]}">{p["name"]}</option>' for p in INITIAL_PRODUCTS.values())
        return SHOP_HTML.replace("__OPTIONS__", options).replace("__CONFIRMED__", word)

    return app


SHOP_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Demo Toy Shop</title></head>
<body>
<h1>Demo Toy Shop</h1>
<label>Product <select id="product">__OPTIONS__</select></label>
<label>Quantity <input id="quantity" type="number" value="1" min="1"></label>
<label>Card <input id="card" value="4242424242424242"></label>
<button id="place-order">Place order</button>
<div id="result" role="status"></div>
<script>
document.getElementById('place-order').addEventListener('click', async () => {
  const result = document.getElementById('result');
  result.textContent = 'Submitting...';
  const response = await fetch('/orders', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer demo-customer-token'},
    body: JSON.stringify({
      product_id: document.getElementById('product').value,
      quantity: Number(document.getElementById('quantity').value),
      payment_card: document.getElementById('card').value
    })
  });
  const body = await response.json();
  result.textContent = response.ok
    ? 'Order __CONFIRMED__: ' + body.id
    : 'Error: ' + (typeof body.detail === 'string' ? body.detail : 'Invalid input');
});
</script>
</body></html>"""

app = create_demo_app()