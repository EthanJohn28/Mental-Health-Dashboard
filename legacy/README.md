# Legacy Model Artifacts

This folder contains archived model files from earlier versions of the Mental Health Dashboard project.

## Purpose
These artifacts are preserved for transparency, comparison, and documentation of the project’s iterative development.  
They are **not** used by the current production dashboard.

## Important Note on Data Leakage
Earlier model versions relied on features that introduced data leakage (e.g., outcome-proxy and post-hoc variables).  
As of **v1.1.0**, the production model was redesigned to use only leakage-safe, pre-observation features to ensure realistic deployment behavior.

## Usage
Do **not** load or deploy these models in production.  
They are retained solely for historical reference and reproducibility.

## Current Model
The active, leakage-safe model and scaler are located in the project root and are used by `streamlit_app.py`.

