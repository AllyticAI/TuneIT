# workers.py
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
import core_logic
import time
import logging
import traceback # For detailed error logging

logger = logging.getLogger(__name__)

# --- Base Worker ---
class WorkerSignals(QObject):
    """Defines signals available from a running worker thread."""
    finished = Signal(object)  # Emits result object on success
    error = Signal(str)        # Emits error message on failure
    progress = Signal(str)     # Emits progress messages/status updates

class BaseWorker(QRunnable):
    """Inheritable worker thread."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add callback to kwargs if necessary (e.g., for training progress)
        # Example: kwargs['progress_callback'] = self.signals.progress.emit

    @Slot()
    def run(self):
        """Execute the worker function."""
        try:
            logger.info(f"Worker started for function: {self.fn.__name__}")
            result = self.fn(*self.args, **self.kwargs)
            logger.info(f"Worker finished successfully for: {self.fn.__name__}")
            self.signals.finished.emit(result)
        except Exception as e:
            error_msg = f"Error in worker ({self.fn.__name__}): {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.signals.error.emit(error_msg)


# --- Specific Workers (Examples - You might need more/different ones) ---

class LoadDataWorker(BaseWorker):
     def __init__(self, file_path):
         super().__init__(core_logic.load_file, file_path)

class PreprocessDataWorker(BaseWorker):
     def __init__(self, df, text_col, label_col, test_size, augment_opts):
         # Combine preprocess and augment for simplicity here
         def preprocess_and_augment(df, text_col, label_col, test_size, augment_opts):
             train_ds, eval_ds = core_logic.preprocess_data(df, text_col, label_col, test_size)
             if augment_opts.get('enabled', False) and train_ds is not None:
                 self.signals.progress.emit("Augmenting training data...") # Emit status
                 train_ds = core_logic.augment_text_data(
                     train_ds,
                     methods=augment_opts.get('methods', []),
                     factor=augment_opts.get('factor', 2)
                 )
             return train_ds, eval_ds

         super().__init__(preprocess_and_augment, df, text_col, label_col, test_size, augment_opts)

class ConfigureModelWorker(BaseWorker):
    def __init__(self, model_name, token, local_only, train_ds, eval_ds, max_len, training_args_dict):
        # Combine loading and tokenizing
        def configure_and_tokenize(model_name, token, local_only, train_ds, eval_ds, max_len, training_args_dict):
            self.signals.progress.emit(f"Loading model and tokenizer: {model_name}...")
            model, tokenizer, msg = core_logic.load_model_with_download(model_name, token, local_only)
            self.signals.progress.emit(f"{msg}. Tokenizing datasets...")
            tokenized_train = core_logic.create_tokenized_dataset(train_ds, tokenizer, max_len)
            tokenized_eval = core_logic.create_tokenized_dataset(eval_ds, tokenizer, max_len)
            self.signals.progress.emit("Tokenization complete.")
            # Return all needed components
            return model, tokenizer, tokenized_train, tokenized_eval, training_args_dict, msg

        super().__init__(configure_and_tokenize, model_name, token, local_only, train_ds, eval_ds, max_len, training_args_dict)


class TrainingWorker(QRunnable): # Custom worker for training to handle specific callback signals
    def __init__(self, model, tokenizer, train_ds, eval_ds, args_dict, callback_emitter):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.args_dict = args_dict
        self.callback_emitter = callback_emitter # The signal emitter instance
        self.signals = WorkerSignals() # Standard finish/error signals

    @Slot()
    def run(self):
        try:
            logger.info("Training worker started.")
            self.callback_emitter.emit_log({"message": "Starting training process..."}) # Use log signal for status too
            trainer, message = core_logic.train_model(
                self.model, self.tokenizer, self.train_ds, self.eval_ds,
                self.args_dict, self.callback_emitter # Pass the emitter here
            )
            logger.info("Training worker finished successfully.")
            self.signals.finished.emit((trainer, message)) # Return trainer and success message
        except Exception as e:
            error_msg = f"Error during training: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.callback_emitter.emit_error(f"Training failed: {e}") # Use specific callback error signal
            self.signals.error.emit(error_msg) # Also emit standard worker error

class ExportWorker(BaseWorker):
     def __init__(self, trainer, export_dir, model_name, formats):
          # Combine export and zip
         def export_and_zip(trainer, export_dir, model_name, formats):
             self.signals.progress.emit("Exporting model formats...")
             exported_paths, final_dir = core_logic.export_model_formats(trainer, export_dir, model_name, formats)
             self.signals.progress.emit(f"Model parts exported to {final_dir}. Creating zip...")
             zip_path = final_dir + ".zip" # Place zip next to the folder
             created_zip = core_logic.create_zip_from_directory(final_dir, zip_path)
             self.signals.progress.emit("Zip file created.")
             return created_zip, exported_paths # Return zip path and dict of exported parts
         super().__init__(export_and_zip, trainer, export_dir, model_name, formats)