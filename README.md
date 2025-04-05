# TuneIT Desktop Fine-Tuning Tool

A desktop application (Windows, macOS, Linux) for fine-tuning Hugging Face Transformer models using your own datasets, built with Python and PySide6.

## Features

*   Load datasets from CSV, JSON, JSONL, TXT files.
*   Preview data.
*   Configure text/label columns and train/test split.
*   Optional basic data augmentation.
*   Select from various pre-trained Hugging Face models (including gated ones via token).
*   Configure common training hyperparameters.
*   Train the model with progress monitoring, logs, and loss charts.
*   Export the fine-tuned model in Hugging Face, PyTorch (.bin), and potentially GGUF (if dependencies installed) formats.
*   Packages exports into a zip file.

## Prerequisites

*   Python 3.8+
*   Git (recommended for installing some dependencies)
*   (Optional but Recommended for Training) NVIDIA GPU with CUDA installed and compatible PyTorch version. See [PyTorch Get Started](https://pytorch.org/get-started/locally/).

## Installation

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone <repository_url>
    cd tuneit_desktop
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **GGUF Export (Optional):** If you need GGUF export, you need `auto-gptq`. Installation can be complex and depend on your CUDA version. Try:
        ```bash
        pip install auto-gptq
        # Or follow instructions from https://github.com/PanQiWei/AutoGPTQ for potentially needed dependencies like ninja.
        ```

4.  **Hugging Face Login (Recommended for Gated Models):**
    If you plan to use models like Llama or Mistral, log in via the terminal *before* running the app, or use the "Set/Update Token" button within the app:
    ```bash
    huggingface-cli login
    ```

## Running the Application

```bash
python main_window.py