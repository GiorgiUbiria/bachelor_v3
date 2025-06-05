# Project Objective

The goal of this project is to demonstrate the integration of machine learning models within a backend-focused e-commerce platform. The system focuses on three key areas where ML adds tangible value to real-world web applications:

## 1. HTTP Request Analysis (Security Monitoring)

Implements an anomaly detection system for incoming HTTP requests to identify and flag potential malicious activity. The model is trained or rule-based to detect 3–4 common web attack patterns:

- Cross-Site Scripting (XSS)
- Cross-Site Request Forgery (CSRF)
- SQL Injection (SQLi)
- Other abnormal request behaviors

## 2. Personalized Recommendations and Dynamic Deals

A recommendation engine analyzes user behavior (views, favorites, comments, region, age group, etc.) to:

- Suggest relevant products in real-time
- Adapt based on trends among users with similar profiles (collaborative filtering)
- Tailor deals and discounts per user cluster

## 3. New Product Automation (Smart Catalog Enrichment)

When a new product is added to the catalog, an ML pipeline automatically:

- Analyzes existing products in the same category
- Matches on descriptions and shared features
- Suggests tags that describe the product semantically (for search and filtering)
- Estimates a price range based on historical pricing trends and similar items