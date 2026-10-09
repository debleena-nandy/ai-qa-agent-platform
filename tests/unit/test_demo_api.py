# File: tests/unit/test_demo_api.py
# Description: Tests order creation, retrieval, and validation in the demo API.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from fastapi.testclient import TestClient

from demo_target.main import create_demo_app

AUTH = {"Authorization": "Bearer demo-customer-token"}
OTHER = {"Authorization": "Bearer demo-other-token"}


def _client(*defects: str) -> TestClient:
    return TestClient(create_demo_app(set(defects)))


def test_demo_api_creates_and_retrieves_order() -> None:
    client = _client()
    created = client.post("/orders", json={"product_id": "toy-001", "quantity": 2}, headers=AUTH)
    assert created.status_code == 201
    order = created.json()
    assert (order["product_id"], order["quantity"], order["status"]) == ("toy-001", 2, "confirmed")
    assert "customer" not in order
    assert client.get(f"/orders/{order['id']}", headers=AUTH).json() == order


def test_demo_api_requires_authentication() -> None:
    assert _client().post("/orders", json={"product_id": "toy-001", "quantity": 1}).status_code == 401


def test_demo_api_rejects_quantity_above_limit() -> None:
    response = _client().post("/orders", json={"product_id": "toy-001", "quantity": 6}, headers=AUTH)
    assert response.status_code == 422


def test_demo_api_hides_other_customers_orders() -> None:
    client = _client()
    order = client.post("/orders", json={"product_id": "toy-001", "quantity": 1}, headers=AUTH).json()
    assert client.get(f"/orders/{order['id']}", headers=OTHER).status_code == 404


def test_demo_api_out_of_stock_and_declined_card() -> None:
    client = _client()
    assert client.post("/orders", json={"product_id": "toy-002", "quantity": 1}, headers=AUTH).status_code == 409
    declined = {"product_id": "toy-001", "quantity": 1, "payment_card": "4000000000000002"}
    assert client.post("/orders", json=declined, headers=AUTH).status_code == 402


def test_seeded_quantity_defect_accepts_six_units() -> None:
    response = _client("quantity_limit").post(
        "/orders", json={"product_id": "toy-001", "quantity": 6}, headers=AUTH)
    assert response.status_code == 201