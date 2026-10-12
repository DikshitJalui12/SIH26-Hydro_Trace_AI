# Contributing to HydroTrace-AI Maritime Command Platform

Thank you for contributing to the **HydroTrace-AI Maritime Command Platform**! We welcome improvements, bug reports, and research enhancements for multi-agent maritime forensics.

---

## Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/DikshitJalui12/SIH26-Hydro_Trace_AI.git
   cd SIH26-Hydro_Trace_AI
   ```

2. **Create a Python Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python -m pytest tests/ -v
   ```

---

## Coding Guidelines

- **Python**: Follow PEP 8 style guidelines. Ensure type hints (`typing`) are present on all public methods and functions.
- **Frontend**: Vanilla CSS & JavaScript without heavyweight dependencies. Zero third-party tracker or paid tile watermark scripts.
- **Security**: Never commit sensitive secrets, test tokens, or private keys. All cryptographic operations must adhere to SHA-256 and HMAC standards.

---

## Testing & Validation

Before submitting any Pull Request:
1. Ensure all existing tests pass:
   ```bash
   python -m pytest tests/ -v
   ```
2. If adding new mathematical models or API endpoints, add corresponding test cases under `tests/`.
3. If modifying model architectures or loss functions, verify convergence by running:
   ```bash
   python train_all_models.py
   ```
