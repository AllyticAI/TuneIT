# main_window.py
import sys
import os
import logging
from pathlib import Path
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableView, QTabWidget, QLineEdit, QComboBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QProgressBar, QTextEdit, QMessageBox, QInputDialog, QListWidget,
    QAbstractItemView, QStatusBar
)
from PySide6.QtCore import (Qt, Slot, QThreadPool, QTimer)
from PySide6.QtGui import (QIcon)
import torch
import transformers # For application icon

# Import local modules
import core_logic
from utils import PandasModel, MplCanvas, CallbackSignalEmitter
from workers import (
    BaseWorker, LoadDataWorker, PreprocessDataWorker, ConfigureModelWorker,
    TrainingWorker, ExportWorker
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TuneIT - Desktop Fine-Tuning Tool")
        # Set window icon (optional)
        # icon_path = Path(__file__).parent / "assets" / "icon.png"
        # if icon_path.exists():
        #     self.setWindowIcon(QIcon(str(icon_path)))

        # --- State Variables (like Streamlit's session_state) ---
        self.raw_dataframe = None
        self.train_dataset = None
        self.eval_dataset = None
        self.model = None
        self.tokenizer = None
        self.tokenized_train = None
        self.tokenized_eval = None
        self.training_args_dict = None
        self.trainer = None
        self.hf_token = core_logic.check_hf_auth() # Check for existing token on startup
        self.current_model_name = ""
        self.output_dir = "" # Base directory for training results

        # Plotting data
        self.train_log_steps = []
        self.train_log_loss = []
        self.eval_log_steps = []
        self.eval_log_loss = []

        # --- Thread Pool ---
        self.threadpool = QThreadPool()
        logger.info(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

        # --- Signal Emitter for Trainer Callback ---
        self.callback_emitter = CallbackSignalEmitter()
        self.setup_callback_connections() # Connect signals here

        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # --- Create Tabs ---
        self.create_upload_tab()
        self.create_configure_tab()
        self.create_train_tab()
        self.create_export_tab()

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a dataset to begin.")

        # --- Set initial size ---
        self.resize(1000, 700)

    # --- UI Creation Methods ---

    def create_upload_tab(self):
        tab_upload = QWidget()
        layout = QVBoxLayout(tab_upload)
        self.tabs.addTab(tab_upload, "1. Upload Dataset")

        # File Selection
        file_layout = QHBoxLayout()
        self.btn_load_file = QPushButton("Load Dataset File (.csv, .json, .jsonl, .txt)")
        self.btn_load_file.clicked.connect(self.select_file)
        self.lbl_file_path = QLabel("No file selected.")
        self.lbl_file_path.setWordWrap(True)
        file_layout.addWidget(self.btn_load_file)
        file_layout.addWidget(self.lbl_file_path, 1) # Stretch label
        layout.addLayout(file_layout)

        # Data Preview
        self.table_preview = QTableView()
        self.table_preview.setAlternatingRowColors(True)
        self.table_preview.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_preview.setWordWrap(False) # Prevent wrapping in table for now
        layout.addWidget(QLabel("Dataset Preview (first 50 rows):"))
        layout.addWidget(self.table_preview, 1) # Stretch table view

        # Preprocessing Options
        preprocess_group = QGroupBox("Preprocessing & Splitting")
        form_layout = QFormLayout(preprocess_group)

        self.combo_text_col = QComboBox()
        self.combo_label_col = QComboBox()
        self.slider_test_size = QSlider(Qt.Orientation.Horizontal)
        self.lbl_test_size = QLabel("20%")
        self.slider_test_size.setRange(5, 50) # 5% to 50%
        self.slider_test_size.setValue(20)
        self.slider_test_size.valueChanged.connect(lambda v: self.lbl_test_size.setText(f"{v}%"))
        test_size_layout = QHBoxLayout()
        test_size_layout.addWidget(self.slider_test_size)
        test_size_layout.addWidget(self.lbl_test_size)

        form_layout.addRow("Text Column:", self.combo_text_col)
        form_layout.addRow("Label Column (Optional):", self.combo_label_col)
        form_layout.addRow("Test Set Size:", test_size_layout)

        # Augmentation Options
        augment_group = QGroupBox("Data Augmentation (Optional)")
        augment_group.setCheckable(True)
        augment_group.setChecked(False)
        augment_layout = QFormLayout(augment_group)
        self.augment_methods = QListWidget()
        self.augment_methods.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.augment_methods.addItems(["Random deletion", "Random swapping", "Synonym replacement (Placeholder)"])
        self.augment_factor_spin = QSpinBox()
        self.augment_factor_spin.setRange(2, 5)
        self.augment_factor_spin.setValue(2)
        augment_layout.addRow("Methods:", self.augment_methods)
        augment_layout.addRow("Factor:", self.augment_factor_spin)

        form_layout.addRow(augment_group) # Add augment group to main form

        layout.addWidget(preprocess_group)

        self.btn_process_data = QPushButton("Process Dataset")
        self.btn_process_data.clicked.connect(self.process_data)
        self.btn_process_data.setEnabled(False) # Enable after loading data
        layout.addWidget(self.btn_process_data)


    def create_configure_tab(self):
        tab_configure = QWidget()
        layout = QVBoxLayout(tab_configure)
        self.tabs.addTab(tab_configure, "2. Configure Model")

        # Model Selection
        model_group = QGroupBox("Base Model Selection")
        model_layout = QFormLayout(model_group)
        self.combo_model_name = QComboBox()
        self.combo_model_name.addItems([
            "gpt2", "gpt2-medium", # "gpt2-large", "gpt2-xl",
            "EleutherAI/gpt-neo-125M", #"EleutherAI/gpt-neo-1.3B",
            "facebook/opt-125m", #"facebook/opt-350m",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # Add gated models - require auth check later
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            # "meta-llama/Llama-2-13b-hf" # Larger models
        ])
        self.combo_model_name.setEditable(True) # Allow custom input
        self.combo_model_name.currentTextChanged.connect(self.check_model_auth_requirement)

        self.check_local_only = QCheckBox("Use only locally cached models (no downloads)")
        self.check_local_only.setChecked(False)

        # Auth Handling
        auth_layout = QHBoxLayout()
        self.lbl_auth_status = QLabel(f"HF Token: {'Detected' if self.hf_token else 'Not Found'}")
        self.btn_set_token = QPushButton("Set/Update Token")
        self.btn_set_token.clicked.connect(self.set_hf_token)
        auth_layout.addWidget(self.lbl_auth_status)
        auth_layout.addWidget(self.btn_set_token)


        model_layout.addRow("Model Name/Path:", self.combo_model_name)
        model_layout.addRow(auth_layout)
        model_layout.addRow(self.check_local_only)
        layout.addWidget(model_group)


        # Hyperparameters
        hyper_group = QGroupBox("Training Hyperparameters")
        hyper_layout = QFormLayout(hyper_group)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(7)
        self.lr_spin.setRange(1e-7, 1e-2)
        self.lr_spin.setValue(5e-5)
        self.lr_spin.setSingleStep(1e-6)

        self.batch_spin = QComboBox() # Use combo for specific powers of 2
        self.batch_spin.addItems([str(2**i) for i in range(0, 8)]) # 1 to 128
        self.batch_spin.setCurrentText("8")

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(3)

        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(-1, 100000) # -1 means use epochs
        self.max_steps_spin.setValue(1000)
        self.max_steps_spin.setSpecialValueText("Use Epochs") # Display text for -1

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 10000)
        self.warmup_spin.setValue(100) # Reduced default

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 64)
        self.grad_accum_spin.setValue(1)

        hyper_layout.addRow("Learning Rate:", self.lr_spin)
        hyper_layout.addRow("Batch Size:", self.batch_spin)
        hyper_layout.addRow("Num Train Epochs:", self.epochs_spin)
        hyper_layout.addRow("Max Steps (-1=ignore):", self.max_steps_spin)
        hyper_layout.addRow("Warmup Steps:", self.warmup_spin)
        hyper_layout.addRow("Gradient Accum. Steps:", self.grad_accum_spin)

        layout.addWidget(hyper_group)

        # Advanced Options
        advanced_group = QGroupBox("Advanced Options")
        advanced_group.setCheckable(True) # Allow hiding
        advanced_group.setChecked(False)
        adv_layout = QFormLayout(advanced_group)

        self.decay_spin = QDoubleSpinBox()
        self.decay_spin.setRange(0.0, 0.5)
        self.decay_spin.setValue(0.01)
        self.decay_spin.setDecimals(4)

        self.check_fp16 = QCheckBox("Use Mixed Precision (FP16/BF16)")
        self.check_fp16.setChecked(torch.cuda.is_available()) # Default true if CUDA available

        self.eval_steps_spin = QSpinBox()
        self.eval_steps_spin.setRange(10, 5000)
        self.eval_steps_spin.setValue(200)
        
        self.save_steps_spin = QSpinBox()
        self.save_steps_spin.setRange(50, 10000)
        self.save_steps_spin.setValue(400)  # Set to multiple of default eval_steps
        
        # Add value change connection to maintain relationship
        self.eval_steps_spin.valueChanged.connect(self.adjust_save_steps)

        self.max_len_spin = QSpinBox()
        self.max_len_spin.setRange(32, 4096) # Increased max range
        self.max_len_spin.setSingleStep(64)
        self.max_len_spin.setValue(512)

        adv_layout.addRow("Weight Decay:", self.decay_spin)
        adv_layout.addRow(self.check_fp16)
        adv_layout.addRow("Evaluation Steps:", self.eval_steps_spin)
        adv_layout.addRow("Save Steps:", self.save_steps_spin)
        adv_layout.addRow("Max Sequence Length:", self.max_len_spin)

        layout.addWidget(advanced_group)

        self.btn_configure_model = QPushButton("Load Model & Prepare Training")
        self.btn_configure_model.clicked.connect(self.configure_model)
        self.btn_configure_model.setEnabled(False) # Enable after processing data
        layout.addWidget(self.btn_configure_model)

        layout.addStretch(1) # Push button to bottom

    def adjust_save_steps(self, new_eval_steps):
        """Adjust save_steps to be a multiple of eval_steps when eval_steps changes."""
        current_save_steps = self.save_steps_spin.value()
        if current_save_steps % new_eval_steps != 0:
            new_save_steps = ((current_save_steps // new_eval_steps) + 1) * new_eval_steps
            self.save_steps_spin.setValue(new_save_steps)
            self.add_log_message(f"Adjusted save_steps to {new_save_steps} to maintain multiple of eval_steps ({new_eval_steps})")

    def create_train_tab(self):
        tab_train = QWidget()
        layout = QVBoxLayout(tab_train)
        self.tabs.addTab(tab_train, "3. Train Model")

        # Start Button
        self.btn_start_train = QPushButton("Start Training")
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_start_train.setEnabled(False) # Enable after configuration
        # Add Stop Button Later
        # self.btn_stop_train = QPushButton("Stop Training")
        # self.btn_stop_train.clicked.connect(self.stop_training)
        # self.btn_stop_train.setEnabled(False)
        layout.addWidget(self.btn_start_train)
        # layout.addWidget(self.btn_stop_train)


        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(QLabel("Training Progress:"))
        layout.addWidget(self.progress_bar)
        self.lbl_progress_steps = QLabel("Step 0 / 0")
        layout.addWidget(self.lbl_progress_steps)


        # Logging Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log_area, 1) # Stretch log area

        # Plotting Area
        plot_layout = QHBoxLayout()
        self.train_plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.eval_plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        plot_layout.addWidget(self.train_plot_canvas)
        plot_layout.addWidget(self.eval_plot_canvas)
        layout.addLayout(plot_layout)


    def create_export_tab(self):
        tab_export = QWidget()
        layout = QVBoxLayout(tab_export)
        self.tabs.addTab(tab_export, "4. Export Model")

        export_group = QGroupBox("Export Settings")
        form = QFormLayout(export_group)

        self.export_formats = QListWidget()
        self.export_formats.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.export_formats.addItems(["huggingface (Required)", "pytorch (.bin)", "gguf (Quantized)"])
        # Select HF by default and make it non-selectable maybe?
        self.export_formats.item(0).setSelected(True)
        self.export_formats.item(0).setFlags(self.export_formats.item(0).flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled) # Make HF mandatory/non-selectable
        # Conditionally enable GGUF if library is present
        if core_logic.AUTO_GPTQ_AVAILABLE:
             self.export_formats.item(2).setFlags(self.export_formats.item(2).flags() | Qt.ItemFlag.ItemIsEnabled)
        else:
             self.export_formats.item(2).setText("gguf (auto-gptq not installed)")
             self.export_formats.item(2).setFlags(self.export_formats.item(2).flags() & ~Qt.ItemFlag.ItemIsEnabled)


        self.export_name_edit = QLineEdit("finetuned-model")

        export_path_layout = QHBoxLayout()
        self.btn_select_export_dir = QPushButton("Select Export Directory")
        self.lbl_export_dir = QLabel("Not selected")
        self.btn_select_export_dir.clicked.connect(self.select_export_directory)
        export_path_layout.addWidget(self.btn_select_export_dir)
        export_path_layout.addWidget(self.lbl_export_dir, 1)

        form.addRow("Export Formats:", self.export_formats)
        form.addRow("Export Model Name:", self.export_name_edit)
        form.addRow("Export Directory:", export_path_layout)

        layout.addWidget(export_group)

        self.btn_export_model = QPushButton("Export Model")
        self.btn_export_model.clicked.connect(self.export_model)
        self.btn_export_model.setEnabled(False) # Enable after training
        layout.addWidget(self.btn_export_model)

        layout.addStretch(1)


    # --- Callback Connections ---
    def setup_callback_connections(self):
        self.callback_emitter.train_begin.connect(self.on_train_begin)
        self.callback_emitter.progress_update.connect(self.update_progress)
        self.callback_emitter.log_received.connect(self.handle_log_message)
        self.callback_emitter.evaluate_received.connect(self.handle_eval_metrics)
        self.callback_emitter.train_end.connect(self.on_train_end)
        self.callback_emitter.error_occurred.connect(self.on_training_error)


    # --- Worker Handling Logic ---

    def run_worker(self, worker_instance):
        """Runs a worker in the thread pool and connects its signals."""
        # Disable buttons maybe? Show busy cursor?
        self.set_ui_busy(True)
        worker_instance.signals.finished.connect(self.on_worker_finished)
        worker_instance.signals.error.connect(self.on_worker_error)
        worker_instance.signals.progress.connect(self.update_status) # Generic progress messages
        self.threadpool.start(worker_instance)

    @Slot(object)
    def on_worker_finished(self, result):
        """Handles successful completion of a worker."""
        self.set_ui_busy(False)
        self.update_status("Task completed successfully.")
        # Here, you need to know WHICH worker finished to process the result
        # This simple example doesn't track worker types. A more robust
        # solution would involve passing context or using different slots.
        # For now, we check the type of result or assume based on workflow.

        # Example: If result is a DataFrame, it's likely from load_data
        if isinstance(result, pd.DataFrame):
            self.handle_data_loaded(result)
        # Example: If result is a tuple (train_ds, eval_ds)
        elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], core_logic.Dataset):
             self.handle_data_processed(result[0], result[1])
        # Example: If result is model config data
        elif isinstance(result, tuple) and len(result) == 6 and isinstance(result[0], torch.nn.Module):
             self.handle_model_configured(*result)
        # Example: Training result
        elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], core_logic.Trainer):
             self.handle_training_complete(*result)
        # Example: Export result
        elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
             self.handle_export_complete(*result)
        else:
             logger.warning(f"Worker finished, but result type not recognized: {type(result)}")


    @Slot(str)
    def on_worker_error(self, error_message):
        """Handles errors reported by workers."""
        self.set_ui_busy(False)
        self.update_status(f"Error: {error_message.splitlines()[0]}", error=True) # Show first line in status
        QMessageBox.critical(self, "Worker Error", f"An error occurred:\n{error_message}")
        # Re-enable potentially disabled buttons depending on context
        self.btn_start_train.setEnabled(self.trainer is None and self.model is not None)
        # etc.

    @Slot(str)
    def update_status(self, message, error=False):
        """Updates the status bar."""
        logger.info(f"Status Update: {message}")
        if error:
             # Maybe add styling later for errors
             self.status_bar.showMessage(f"Error: {message}")
             # Add to log area as well
             self.log_area.append(f"<font color='red'><b>[ERROR]</b> {message}</font>")
        else:
            self.status_bar.showMessage(message)
            # Also add non-error status updates to log for history? Optional.
            # self.log_area.append(f"[INFO] {message}")


    def set_ui_busy(self, busy):
        """Enable/disable UI elements during processing."""
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            # Disable relevant buttons
            self.btn_load_file.setEnabled(False)
            self.btn_process_data.setEnabled(False)
            self.btn_configure_model.setEnabled(False)
            self.btn_start_train.setEnabled(False)
            self.btn_export_model.setEnabled(False)
        else:
            QApplication.restoreOverrideCursor()
            # Re-enable based on state
            self.btn_load_file.setEnabled(True)
            self.btn_process_data.setEnabled(self.raw_dataframe is not None)
            self.btn_configure_model.setEnabled(self.train_dataset is not None)
            # Only enable train if model is configured AND training not running/complete
            is_trainable = self.model is not None and self.trainer is None # Basic check
            self.btn_start_train.setEnabled(is_trainable)
            self.btn_export_model.setEnabled(self.trainer is not None)

    # --- Action Handlers ---

    @Slot()
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset File", "",
            "Dataset Files (*.csv *.json *.jsonl *.txt);;All Files (*)"
        )
        if file_path:
            self.lbl_file_path.setText(file_path)
            self.update_status(f"Loading data from: {file_path}")
            worker = LoadDataWorker(file_path)
            self.run_worker(worker)

    def handle_data_loaded(self, df):
        self.raw_dataframe = df
        self.update_status(f"Loaded {len(df)} rows. Configure preprocessing.")
        # Update preview table
        preview_df = df.head(50) # Show only first 50 rows
        model = PandasModel(preview_df)
        self.table_preview.setModel(model)
        self.table_preview.resizeColumnsToContents()
        # Update column selectors
        columns = [""] + df.columns.tolist() # Add empty option for label
        self.combo_text_col.clear()
        self.combo_text_col.addItems(df.columns.tolist())
        self.combo_label_col.clear()
        self.combo_label_col.addItems(columns)
        # Try to guess default text column
        for col in ['text', 'content', 'prompt', 'input']:
             if col in df.columns:
                  self.combo_text_col.setCurrentText(col)
                  break
        self.btn_process_data.setEnabled(True)
        # Reset downstream state if data is reloaded
        self.reset_downstream_state('data_loaded')


    @Slot()
    def process_data(self):
        if self.raw_dataframe is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        text_col = self.combo_text_col.currentText()
        label_col = self.combo_label_col.currentText()
        if not label_col: # Handle empty selection
            label_col = None
        test_size = self.slider_test_size.value() / 100.0

        augment_opts = {
            "enabled": self.augment_methods.parent().isChecked(), # Check if groupbox is checked
            "methods": [item.text() for item in self.augment_methods.selectedItems()],
            "factor": self.augment_factor_spin.value()
        }

        self.update_status("Processing and splitting data...")
        worker = PreprocessDataWorker(self.raw_dataframe, text_col, label_col, test_size, augment_opts)
        self.run_worker(worker)

    def handle_data_processed(self, train_ds, eval_ds):
        self.train_dataset = train_ds
        self.eval_dataset = eval_ds
        self.update_status(f"Data processed. Train: {len(train_ds)}, Eval: {len(eval_ds)}. Configure model.")
        self.btn_configure_model.setEnabled(True)
        # Reset downstream state
        self.reset_downstream_state('data_processed')


    @Slot()
    def set_hf_token(self):
        token, ok = QInputDialog.getText(self, "Hugging Face Token",
                                          "Enter your Hugging Face API token (read/write permissions recommended):",
                                          QLineEdit.EchoMode.Password) # Use Password mode
        if ok and token:
            # Validate and save the token
            self.update_status("Validating token...")
            QApplication.processEvents() # Update UI
            try:
                 success, msg = core_logic.login_huggingface(token)
                 if success:
                      self.hf_token = token
                      self.lbl_auth_status.setText(f"HF Token: Detected ({msg.split(':')[-1].strip()})")
                      self.update_status("Hugging Face token updated and verified.")
                      QMessageBox.information(self, "Token Set", f"Token accepted. {msg}")
                 else:
                      self.update_status(f"Token validation failed: {msg}", error=True)
                      QMessageBox.warning(self, "Token Invalid", f"The provided token seems invalid.\n{msg}")
            except Exception as e:
                 self.update_status(f"Error setting token: {e}", error=True)
                 QMessageBox.critical(self, "Error", f"An error occurred while setting the token:\n{e}")
        elif ok: # User pressed OK but entered no token
             self.hf_token = None
             self.lbl_auth_status.setText("HF Token: Removed")
             self.update_status("Hugging Face token removed.")


    @Slot(str)
    def check_model_auth_requirement(self, model_name):
         # Simple check for known gated model patterns
         is_gated = any(pattern in model_name.lower() for pattern in ['llama', 'mistral', 'gemma'])
         if is_gated and not self.hf_token:
              self.lbl_auth_status.setText("HF Token: <font color='red'>Required for this model!</font>")
         elif self.hf_token:
              # Reset message if token exists (re-validation happens on load)
               self.lbl_auth_status.setText(f"HF Token: Detected") # Simplified status
         else:
              self.lbl_auth_status.setText("HF Token: Not Found")


    @Slot()
    def configure_model(self):
        if self.train_dataset is None or self.eval_dataset is None:
            QMessageBox.warning(self, "Data Missing", "Please process data first.")
            return

        model_name = self.combo_model_name.currentText()
        local_only = self.check_local_only.isChecked()
        max_len = self.max_len_spin.value()

        # Check auth for gated models
        is_gated = any(pattern in model_name.lower() for pattern in ['llama', 'mistral', 'gemma'])
        if is_gated and not self.hf_token:
             QMessageBox.critical(self, "Authentication Required",
                                 f"Model '{model_name}' likely requires a Hugging Face token. "
                                 "Please set one using the 'Set/Update Token' button.")
             return

        # --- Gather Training Args ---
        eval_steps = self.eval_steps_spin.value()
        save_steps = self.save_steps_spin.value()
        
        # Adjust save_steps to be a multiple of eval_steps
        if save_steps % eval_steps != 0:
            original_save_steps = save_steps
            save_steps = ((save_steps // eval_steps) + 1) * eval_steps
            self.save_steps_spin.setValue(save_steps)  # Update the UI
            self.add_log_message(f"Adjusted save_steps from {original_save_steps} to {save_steps} to be multiple of eval_steps ({eval_steps})")

        # Ensure output dir exists and is unique per run maybe?
        base_output_dir = Path("./training_results") # Store results locally
        run_name = f"{model_name.split('/')[-1]}-finetuned-{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        self.output_dir = base_output_dir / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training output directory set to: {self.output_dir}")

        self.training_args_dict = {
            "output_dir": str(self.output_dir),
            "learning_rate": self.lr_spin.value(),
            "per_device_train_batch_size": int(self.batch_spin.currentText()),
            "per_device_eval_batch_size": int(self.batch_spin.currentText()), # Use same for eval
            "num_train_epochs": self.epochs_spin.value(),
            "max_steps": self.max_steps_spin.value(),
            "warmup_steps": self.warmup_spin.value(),
            "gradient_accumulation_steps": self.grad_accum_spin.value(),
            "weight_decay": self.decay_spin.value(),
            "logging_dir": str(self.output_dir / "logs"),
            "logging_steps": 50, # Log frequently for UI updates
            "evaluation_strategy": "steps",
            "eval_steps": eval_steps,
            "save_strategy": "steps", # Match eval strategy often
            "save_steps": save_steps,  # Use adjusted value
            "save_total_limit": 2,  # Keep only last 2 checkpoints
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss", # Monitor eval loss
            "greater_is_better": False,          # Lower eval loss is better
            "report_to": "none", # Disable external reporting (wandb, etc.)
            # Handle fp16/bf16 based on checkbox and availability
            "fp16": self.check_fp16.isChecked() and torch.cuda.is_available() and torch.cuda.is_bf16_supported() == False,
            "bf16": self.check_fp16.isChecked() and torch.cuda.is_available() and torch.cuda.is_bf16_supported() == True,
        }
        # Clean args: remove fp16/bf16 if False
        if not self.training_args_dict["fp16"]: del self.training_args_dict["fp16"]
        if not self.training_args_dict["bf16"]: del self.training_args_dict["bf16"]


        self.update_status(f"Configuring model: {model_name}...")
        worker = ConfigureModelWorker(
            model_name, self.hf_token, local_only,
            self.train_dataset, self.eval_dataset, max_len,
            self.training_args_dict # Pass the dict here
        )
        self.run_worker(worker)

    def handle_model_configured(self, model, tokenizer, tok_train, tok_eval, args_dict, msg):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenized_train = tok_train
        self.tokenized_eval = tok_eval
        self.training_args_dict = args_dict # Get updated dict back (output_dir is now set)
        self.current_model_name = model.config._name_or_path # Store loaded name

        total_params = sum(p.numel() for p in model.parameters()) / 1_000_000 # In Millions
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
        self.update_status(f"Model '{self.current_model_name}' loaded ({trainable_params:.2f}M / {total_params:.2f}M params). Ready for training.")
        self.add_log_message(f"Configuration successful: {msg}")
        self.add_log_message(f"Model: {self.current_model_name}")
        self.add_log_message(f"Params (Trainable/Total): {trainable_params:.2f}M / {total_params:.2f}M")
        self.add_log_message(f"Tokenized Train examples: {len(tok_train)}, Eval examples: {len(tok_eval)}")
        self.add_log_message(f"Training args: {args_dict}")

        self.btn_start_train.setEnabled(True)
        # Reset downstream state
        self.reset_downstream_state('model_configured')


    @Slot()
    def start_training(self):
        if not all([self.model, self.tokenizer, self.tokenized_train, self.tokenized_eval, self.training_args_dict]):
             QMessageBox.critical(self, "Error", "Model or data not configured properly.")
             return

        self.add_log_message("--- Starting Training Run ---")
        # Reset plots and progress
        self.progress_bar.setValue(0)
        self.lbl_progress_steps.setText("Step 0 / ?")
        self.train_log_steps.clear()
        self.train_log_loss.clear()
        self.eval_log_steps.clear()
        self.eval_log_loss.clear()
        self.train_plot_canvas.clear_plot()
        self.eval_plot_canvas.clear_plot()

        self.trainer = None # Reset trainer state before starting
        self.btn_export_model.setEnabled(False)

        # Use the custom TrainingWorker
        worker = TrainingWorker(
             self.model, self.tokenizer, self.tokenized_train, self.tokenized_eval,
             self.training_args_dict, self.callback_emitter # Pass the emitter
        )
        # Training worker uses different signals, connect them
        worker.signals.finished.connect(self.on_worker_finished) # Standard finish
        worker.signals.error.connect(self.on_worker_error)       # Standard error

        # We don't use the generic progress signal here, rely on callback signals
        self.update_status("Starting training...")
        self.set_ui_busy(True) # Manually set busy, worker doesn't use generic progress signal
        self.threadpool.start(worker)


    def handle_training_complete(self, trainer, message):
         # Note: This is called from the *standard* worker finish signal
         self.trainer = trainer
         self.update_status(f"Training finished: {message}")
         self.add_log_message(f"--- Training Complete: {message} ---")
         self.btn_export_model.setEnabled(True)
         self.set_ui_busy(False) # Ensure UI is re-enabled


    @Slot()
    def select_export_directory(self):
         dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
         if dir_path:
              self.lbl_export_dir.setText(dir_path)

    @Slot()
    def export_model(self):
        if self.trainer is None:
            QMessageBox.warning(self, "No Trained Model", "Please train a model first.")
            return

        export_dir = self.lbl_export_dir.text()
        if not export_dir or export_dir == "Not selected":
            QMessageBox.warning(self, "No Directory", "Please select an export directory.")
            return

        model_name = self.export_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid name for the exported model.")
            return

        selected_items = self.export_formats.selectedItems()
        formats = [item.text().split(" ")[0] for item in selected_items] # Get format name (e.g., 'pytorch')
        if 'huggingface' not in formats: # Ensure base format is always included
             formats.append('huggingface')
             # Find and select the HF item visually if needed (optional)
             for i in range(self.export_formats.count()):
                  if "huggingface" in self.export_formats.item(i).text():
                       self.export_formats.item(i).setSelected(True)
                       break

        # GGUF specific check - re-verify library as sanity check
        if 'gguf' in formats and not core_logic.AUTO_GPTQ_AVAILABLE:
            QMessageBox.warning(self,"GGUF Unavailable", "GGUF export was selected, but the required 'auto_gptq' library is not installed or failed to import.")
            formats.remove('gguf') # Remove it from the list
            if not formats or formats == ['huggingface']: # Check if only HF is left or empty
                 self.add_log_message("Skipping export as only unavailable/required formats selected.")
                 return # Stop if nothing valid is left to export

        self.update_status(f"Exporting model '{model_name}' to {export_dir}...")
        worker = ExportWorker(self.trainer, export_dir, model_name, formats)
        self.run_worker(worker)


    def handle_export_complete(self, zip_path, exported_paths):
         self.update_status(f"Model exported and zipped to {zip_path}")
         self.add_log_message(f"--- Export Complete ---")
         self.add_log_message(f"Model saved in formats: {list(exported_paths.keys())}")
         self.add_log_message(f"Output zipped to: {zip_path}")
         QMessageBox.information(self, "Export Complete", f"Model successfully exported and zipped to:\n{zip_path}")


    # --- Trainer Callback Slots ---

    @Slot(int)
    def on_train_begin(self, max_steps):
        self.progress_bar.setRange(0, max_steps if max_steps > 0 else 100) # Use 100 if steps unknown (-1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"%p% - Step 0 / {max_steps if max_steps > 0 else '?'}")
        self.lbl_progress_steps.setText(f"Step 0 / {max_steps if max_steps > 0 else '?'}")
        self.add_log_message(f"Training started. Max steps: {max_steps if max_steps > 0 else 'Epoch-based'}")

    @Slot(int, int, float)
    def update_progress(self, step, max_steps, progress):
        # Update progress bar value
        current_max = max_steps if max_steps > 0 else 1 # Avoid division by zero for format string
        self.progress_bar.setValue(step)
        self.progress_bar.setFormat(f"{progress * 100:.1f}% - Step {step} / {current_max}")
        self.lbl_progress_steps.setText(f"Step {step} / {current_max}")


    @Slot(dict)
    def handle_log_message(self, log_data):
        step = log_data.get('step', self.train_log_steps[-1] if self.train_log_steps else 0) # Get step, fallback needed
        log_str = f"[Step {step}]"
        if 'loss' in log_data:
            loss = log_data['loss']
            log_str += f" Train Loss: {loss:.4f}"
            self.train_log_steps.append(step)
            self.train_log_loss.append(loss)
            # Limit history size if needed
            # if len(self.train_log_steps) > 1000: ...
            self.train_plot_canvas.plot(self.train_log_steps, self.train_log_loss, "Training Loss", "Step", "Loss", "Train")
        if 'learning_rate' in log_data:
            lr = log_data['learning_rate']
            log_str += f" LR: {lr:.2e}" # Scientific notation for LR
        if 'epoch' in log_data:
             epoch = log_data['epoch']
             log_str += f" Epoch: {epoch:.2f}"
        if 'message' in log_data: # For custom status messages
             log_str = f"[INFO] {log_data['message']}"

        self.add_log_message(log_str)

    @Slot(dict)
    def handle_eval_metrics(self, metrics):
        step = metrics.get('step', self.eval_log_steps[-1] if self.eval_log_steps else 0) # Get step
        log_str = f"[Step {step} Eval]"
        if 'eval_loss' in metrics:
            loss = metrics['eval_loss']
            log_str += f" Eval Loss: {loss:.4f}"
            self.eval_log_steps.append(step)
            self.eval_log_loss.append(loss)
            self.eval_plot_canvas.plot(self.eval_log_steps, self.eval_log_loss, "Evaluation Loss", "Step", "Loss", "Eval")
        # Add other eval metrics if logged (e.g., accuracy, perplexity)
        for key, value in metrics.items():
            if key not in ['eval_loss', 'step', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second']:
                log_str += f" {key}: {value:.4f}"

        self.add_log_message(log_str)

    @Slot(str)
    def on_train_end(self, message):
         # This signal might be redundant if the main worker finish signal is used
         # but can be useful for final log messages specific to training completion.
         self.add_log_message(f"[INFO] Training sequence ended: {message}")
         # Don't re-enable UI here, let the main worker finish signal handle that


    @Slot(str)
    def on_training_error(self, error_message):
         # This signal comes specifically from the callback/trainer internals
         self.add_log_message(f"<font color='red'><b>[TRAINING ERROR]</b> {error_message}</font>")
         # Don't necessarily stop the whole worker here, let it propagate
         # UI re-enabling should happen in the main worker error slot


    # --- Utility Methods ---
    def add_log_message(self, message):
        """Appends a message to the log area, handling potential HTML."""
        # Simple check for basic html tags to avoid double-escaping
        if message.startswith("<font") or message.startswith("["):
             self.log_area.append(message)
        else:
             self.log_area.append(f"[INFO] {message}") # Add default prefix if plain text
        # Auto-scroll to the bottom
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())


    def reset_downstream_state(self, stage):
        """Resets state variables when an earlier stage is redone."""
        logger.info(f"Resetting state from stage: {stage}")
        if stage == 'data_loaded':
            self.train_dataset = None
            self.eval_dataset = None
            self.tokenized_train = None
            self.tokenized_eval = None
            self.btn_configure_model.setEnabled(False)
            # Fall through to reset model state too
            stage = 'data_processed'

        if stage == 'data_processed':
             self.model = None
             self.tokenizer = None
             self.tokenized_train = None
             self.tokenized_eval = None
             self.training_args_dict = None
             self.btn_start_train.setEnabled(False)
             # Fall through
             stage = 'model_configured'

        if stage == 'model_configured':
             self.trainer = None
             self.btn_export_model.setEnabled(False)
             # Reset training progress UI
             self.progress_bar.setValue(0)
             self.progress_bar.setFormat("%p%")
             self.lbl_progress_steps.setText("Step 0 / 0")
             self.log_area.clear()
             self.train_plot_canvas.clear_plot()
             self.eval_plot_canvas.clear_plot()
             self.train_log_steps.clear()
             self.train_log_loss.clear()
             self.eval_log_steps.clear()
             self.eval_log_loss.clear()

    def closeEvent(self, event):
        """Ensure threads are cleaned up on exit."""
        logger.info("Close event triggered. Waiting for threads to finish...")
        # Note: QThreadPool automatically manages threads, but you might
        # want to implement cancellation logic for active workers if needed.
        # For simplicity, we just wait briefly.
        self.threadpool.waitForDone(2000) # Wait max 2 seconds
        logger.info("Threads finished or timeout reached. Exiting.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())