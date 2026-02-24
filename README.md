<p align="center">
  <h1 align="center">SCOPE & REACH</h1>
  <p align="center">
    <strong>Optimized Inference for Generative Risk Scoring</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/status-active_development-yellow.svg" alt="Status: Active Development">
    <!-- Uncomment / customize these as needed:
    <a href="https://pypi.org/project/your-package/"><img src="https://img.shields.io/pypi/v/your-package.svg" alt="PyPI"></a>
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
    <a href="https://github.com/yourusername/yourrepo/actions"><img src="https://github.com/yourusername/yourrepo/workflows/Tests/badge.svg" alt="Tests"></a>
    -->
  </p>
</p>

---

A general-purpose Python package providing easy-to-use implementations of the **SCOPE** and **REACH** estimators for efficient generative risk score prediction.

## 🚀 Quick Start

All you need to get started with generative risk scoring:

| Requirement | Description |
|---|---|
| **Model** | Any [SGLang](https://github.com/sgl-project/sglang)-compatible model |
| **Sequences** | Tokenized sequences as a `list[list[int]]` |
| **Outcome token** | The token ID representing the outcome of interest |
| **Suppressed tokens** | Token IDs to suppress during generation (e.g., padding tokens) |
| **Stop tokens** | Token IDs that signal the end of a timeline |

## 📦 Installation

```bash
pip install -e SCOPE_REACH_optimized_inference
```

## ⏱️ Time-Based Termination (WIP)

There is work-in-progress support for time-based termination. Currently this supports time-spacing tokens only. Future updates will generalize this further and allow users to define custom termination logic.


## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
