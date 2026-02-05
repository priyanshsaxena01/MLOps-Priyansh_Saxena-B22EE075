# Assignment 2: Data Contracts & YAML

**Name:** Priyansh Saxena
**Roll Number:** B22EE075

## Overview
This submission contains four (4) distinct data contracts in YAML format. These contracts are designed to act as stable interfaces between data producers (operational databases, IoT sensors) and data consumers (ML models, dashboards), preventing downstream breakages caused by schema changes or data quality issues.

All YAML files have been validated for syntax correctness.

## File Manifest

### 1. `rides_contract.yaml` (Scenario 1: Ride-Share)
**Context:** A comprehensive contract for CityMove's ride-sharing platform to protect the Dynamic Pricing Algorithm.
* **Logical Mapping:** Renamed cryptic columns (e.g., `ts_start` -> `pickup_timestamp`, `fare_final` -> `fare_amount_usd`) to business-friendly names.
* **Quality Rules:**
    * Fare must be non-negative.
    * Driver rating must be between 1.0 and 5.0.
    * Distance must not be null.
* **Governance:** `passenger_id` is flagged as PII.
* **SLA:** Freshness threshold set to 30 minutes for hourly model retraining.
* **Negotiation:** Includes a summary of the agreement between the Data Engineering and ML Teams.

### 2. `orders_contract.yaml` (Scenario 2: E-commerce)
**Context:** A real-time order stream for a Black Friday marketing dashboard.
* **Schema Enum:** Maps raw status codes (2, 5, 9) to logical string values (`PAID`, `SHIPPED`, `CANCELLED`).
* **Quality Rules:**
    * Order total must be >= 0.
    * **Circuit Breaker:** Rejects any records with unmapped status codes (e.g., legacy code 7) to prevent dashboard crashes.

### 3. `thermostat_contract.yaml` (Scenario 3: IoT)
**Context:** Telemetry from 10,000 smart thermostats used for average home temperature reporting.
* **Temperature Check:** Enforces range of -30C to 60C to filter out sensor malfunction defaults (e.g., 9999C).
* **Battery Check:** Enforces battery level range between 0.0 and 1.0 (inclusive).

### 4. `fintech_contract.yaml` (Scenario 4: FinTech)
**Context:** A transaction log for a bank's fraud detection system requiring strict ID formatting.
* **Pattern Matching:** Uses Regex to enforce `source_account_id` as exactly 10 uppercase alphanumeric characters.
* **Enforcement:** Explicitly annotated as a **Hard Circuit Breaker** to block the pipeline upon violation, preventing silent failures in the fraud model.

## Tools Used
* **Validation:** Syntax verified using yamllint.com.
* **Standard:** Based on the Open Data Contract Standard (ODCS) pattern provided in the assignment template.
