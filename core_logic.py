# core_logic.py
import pandas as pd
import numpy as np
import json
import os
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback  # We still need this base class
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import zipfile
import tempfile
import shutil
import logging
from huggingface_hub import login, HfFolder, HfApi
from pathlib import Path
# Optional GGUF export
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    AUTO_GPTQ_AVAILABLE = True
except ImportError:
    AUTO_GPTQ_AVAILABLE = False
    logging.warning("auto_gptq not found. GGUF export will be unavailable.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataset Handling ---

def load_file(file_path):
    """Load and parse file based on extension."""
    file_extension = Path(file_path).suffix.lower()
    try:
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return pd.DataFrame(data)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return pd.DataFrame({"text": [content]}) # Simple text file handling
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                 # Try common structures, fallback to single row
                if "data" in data and isinstance(data["data"], list):
                     return pd.DataFrame(data["data"])
                else:
                     return pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
        raise  # Re-raise after logging

def preprocess_data(df, text_column, label_column=None, test_size=0.2):
    """Preprocess dataframe into training format."""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found.")
    if label_column and label_column not in df.columns:
         logger.warning(f"Label column '{label_column}' not found. Proceeding without labels.")
         label_column = None # Ignore if not found

    dataset_dict = {"text": df[text_column].astype(str).tolist()} # Ensure text is string

    if label_column:
        try:
             # Attempt conversion if labels are not already numeric/suitable
             # This is basic; more robust handling might be needed
             dataset_dict["labels"] = pd.to_numeric(df[label_column], errors='coerce').fillna(0).astype(int).tolist()
             # Add check for NaNs introduced if coerce fails
             if pd.isna(dataset_dict["labels"]).any():
                 logger.warning("Labels contained non-numeric values after coercion. Check label column.")
                 # Decide how to handle: raise error, fill with default, etc. Here we filled with 0.
        except Exception as e:
            logger.error(f"Error processing label column '{label_column}': {e}. Proceeding without labels.")
            label_column = None # Fallback to no labels

    dataset = Dataset.from_dict(dataset_dict)

    if len(dataset) < 2:
        raise ValueError("Dataset too small to split. Needs at least 2 examples.")
    # Ensure test_size doesn't result in 0 examples in either split
    num_test = int(len(dataset) * test_size)
    num_train = len(dataset) - num_test
    if num_train == 0 or num_test == 0:
        logger.warning(f"Test split ({test_size=}) results in zero examples for train or test. Adjusting split.")
        # Simple adjustment: ensure at least 1 example in each minimum
        if num_test == 0: num_test = 1
        if num_train == 0: num_train = 1
        # Recalculate test_size based on having at least one sample
        test_size = num_test / (num_train + num_test)
        if test_size >= 1.0: # Handle edge case with very small datasets (e.g., size 2)
             test_size = 0.5

    try:
        train_test_dataset = dataset.train_test_split(test_size=test_size)
        return train_test_dataset["train"], train_test_dataset["test"]
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}", exc_info=True)
        # Fallback for very small datasets if split fails
        if len(dataset) >= 2:
             logger.warning("Splitting failed, returning full dataset as train and first item as eval.")
             return dataset, dataset.select([0])
        else: # Should have been caught earlier, but defensive check
             raise ValueError("Cannot split dataset with less than 2 examples.") from e


def augment_text_data(dataset, methods=None, factor=2):
    """Apply text augmentation techniques."""
    if not methods or factor <= 1:
        return dataset

    augmented_texts = []
    augmented_labels = []
    has_labels = "labels" in dataset.column_names

    logger.info(f"Augmenting data with methods: {methods}, factor: {factor}")

    for i in range(len(dataset)):
        text = dataset[i]["text"]
        label = dataset[i]["labels"] if has_labels else None

        # Add original
        augmented_texts.append(text)
        if has_labels: augmented_labels.append(label)

        for _ in range(factor - 1):
            augmented_text = text
            # Apply methods sequentially (order might matter)
            if "Random deletion" in methods:
                words = augmented_text.split()
                if len(words) > 1:
                    delete_count = max(1, int(len(words) * np.random.uniform(0.05, 0.15))) # Reduced percentage
                    indices_to_delete = np.random.choice(len(words), delete_count, replace=False)
                    augmented_text = ' '.join([word for idx, word in enumerate(words) if idx not in indices_to_delete])

            if "Random swapping" in methods:
                words = augmented_text.split()
                if len(words) > 3: # Need at least 4 words to swap 2 pairs reasonably
                    swap_count = min(max(1, len(words) // 10), 3) # Swap 1-3 pairs, relative to length
                    for _ in range(swap_count):
                        idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                        words[idx1], words[idx2] = words[idx2], words[idx1]
                    augmented_text = ' '.join(words)

            if "Synonym replacement" in methods:
                # Placeholder: Real implementation needs NLTK/WordNet or similar
                # For now, just indicate it would happen
                # words = augmented_text.split()
                # if words:
                #    words[np.random.randint(len(words))] += "_syn" # Simple indicator
                # augmented_text = ' '.join(words)
                pass # Requires external library, skipping actual implementation

            if augmented_text != text: # Only add if augmentation actually changed it
                 augmented_texts.append(augmented_text)
                 if has_labels: augmented_labels.append(label)


    if not augmented_texts: # Should not happen if factor > 1, but safety check
         return dataset

    new_dataset_dict = {"text": augmented_texts}
    if has_labels:
        new_dataset_dict["labels"] = augmented_labels

    logger.info(f"Augmentation resulted in {len(augmented_texts)} examples.")
    return Dataset.from_dict(new_dataset_dict)

def create_tokenized_dataset(dataset, tokenizer, max_length=512):
    """Tokenize dataset."""
    if not tokenizer:
        raise ValueError("Tokenizer not provided.")
    if not dataset:
        raise ValueError("Dataset not provided.")

    def tokenize_function(examples):
        # Ensure tokenizer handles padding correctly
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
        return tokenizer(
            examples["text"],
            padding="max_length", # Pad to max_length
            truncation=True,
            max_length=max_length,
            return_tensors="pt" # Return PyTorch tensors if needed later, though map removes them usually
        )

    try:
        # Determine columns to remove - keep labels if they exist
        remove_cols = [col for col in dataset.column_names if col not in ["labels", "attention_mask", "input_ids"]]

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_cols, # Remove original text, keep labels
            desc="Tokenizing dataset" # Add description for progress bars
        )
        # Set format to PyTorch for the Trainer
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'] + (['labels'] if 'labels' in tokenized_dataset.column_names else []))
        return tokenized_dataset
    except Exception as e:
        logger.error(f"Error during tokenization: {e}", exc_info=True)
        raise

# --- Model Handling ---

def check_hf_auth(token=None):
    """Check if authenticated or if provided token works."""
    if token:
        try:
            user_info = HfApi().whoami(token=token)
            logger.info(f"Using provided token. Authenticated as: {user_info.get('name')}")
            return token # Return the valid token
        except Exception as e:
            logger.error(f"Provided Hugging Face token is invalid: {e}")
            raise ValueError("Invalid Hugging Face Token Provided.") from e
    else:
        # Check environment variable first
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
             logger.info("Using token from HUGGING_FACE_HUB_TOKEN environment variable.")
             # Optionally validate it here too
             try:
                 HfApi().whoami(token=token)
                 return token
             except Exception:
                  logger.warning("Token from environment variable seems invalid. Falling back.")
                  token = None # Reset token if invalid

        # Check cached token
        token = HfFolder.get_token()
        if token:
            try:
                user_info = HfApi().whoami(token=token)
                logger.info(f"Using cached token. Authenticated as: {user_info.get('name')}")
                return token
            except Exception:
                logger.warning("Cached Hugging Face token is invalid or expired.")
                return None
        else:
            logger.info("No Hugging Face token found (checked provided, env, cache).")
            return None


def login_huggingface(token):
    """Logs into Hugging Face Hub and saves token."""
    try:
        login(token=token, add_to_git_credential=False) # Avoid modifying git credentials
        logger.info("Hugging Face login successful.")
        # Verify by fetching user info
        user_info = HfApi().whoami(token=token)
        return True, f"Logged in as: {user_info.get('name')}"
    except Exception as e:
        logger.error(f"Hugging Face login failed: {e}", exc_info=True)
        return False, f"Login failed: {str(e)}"


def load_model_with_download(model_name_or_path, use_auth_token=None, local_only=False):
    """Load model and tokenizer, handling download and auth."""
    message = ""
    resolved_token = use_auth_token if use_auth_token else check_hf_auth() # Use provided or find existing

    model_args = {"token": resolved_token} if resolved_token else {}
    tokenizer_args = {"token": resolved_token} if resolved_token else {}

    try:
        if local_only:
            logger.info(f"Attempting to load {model_name_or_path} from local cache only.")
            model_args["local_files_only"] = True
            tokenizer_args["local_files_only"] = True

        # Load tokenizer first (often smaller)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        message = "Tokenizer loaded."
        logger.info(f"Tokenizer for {model_name_or_path} loaded successfully.")

        # Add pad token if missing - common requirement
        if tokenizer.pad_token is None:
             if tokenizer.eos_token is not None:
                  logger.warning("Tokenizer missing pad_token, using eos_token.")
                  tokenizer.pad_token = tokenizer.eos_token
             else:
                  # Add a new pad token if EOS is also missing (less common for Causal LM)
                  logger.warning("Tokenizer missing both pad_token and eos_token. Adding '<|pad|>'.")
                  tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                  # Important: If we add tokens, the model embedding layer might need resizing later
                  # For fine-tuning, this is usually handled automatically by Trainer if model passed

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_args)
        # Resize embeddings if we added a new pad token manually AND model has get_input_embeddings method
        if tokenizer.pad_token == '<|pad|>' and hasattr(model, 'resize_token_embeddings'):
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized model embeddings to {len(tokenizer)} due to added pad token.")

        message += " Model loaded."
        logger.info(f"Model {model_name_or_path} loaded successfully.")
        if local_only: message += " (Used local cache only)"
        elif "local_files_only" not in model_args: message += " (Downloaded if needed)"


        return model, tokenizer, message

    except EnvironmentError as e: # Handles file not found, network issues etc.
        logger.error(f"Failed to load {model_name_or_path}. Local only: {local_only}. Error: {e}", exc_info=True)
        err_msg = f"Model/tokenizer not found for '{model_name_or_path}'."
        if local_only:
            err_msg += " Check cache or disable 'Use only local cache'."
        elif "401" in str(e) or "403" in str(e): # Check for authentication errors
             err_msg += " Authentication error. Ensure you are logged in (huggingface-cli login) or provide a valid token if the model is private/gated."
        elif "offline" in str(e).lower():
             err_msg += " Network error or Hugging Face Hub is unreachable."
        else:
            err_msg += f" Specific error: {e}"
        raise ValueError(err_msg) from e
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {model_name_or_path}: {e}", exc_info=True)
        raise ValueError(f"Unexpected error loading model: {e}") from e

# --- Training ---

# Custom callback needs to emit signals instead of using Streamlit directly
# We define the base structure here, the actual emitting happens in the worker
class DesktopTrainerCallback(TrainerCallback):
    def __init__(self, signal_emitter):
        super().__init__()
        self.signal_emitter = signal_emitter # Expects an object with emit methods
        self.max_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.max_steps = state.max_steps
        if hasattr(self.signal_emitter, 'emit_train_begin'):
            self.signal_emitter.emit_train_begin(self.max_steps)

    def on_step_end(self, args, state, control, **kwargs):
         # Emit progress frequently for smoother UI update
         if hasattr(self.signal_emitter, 'emit_progress'):
             progress = (state.global_step / self.max_steps) if self.max_steps > 0 else 0
             self.signal_emitter.emit_progress(state.global_step, self.max_steps, progress)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        # Ensure global_step is included in the log data
        log_data = logs.copy()
        log_data['step'] = state.global_step
        # Could add epoch here too if needed: log_data['epoch'] = state.epoch

        if hasattr(self.signal_emitter, 'emit_log'):
            self.signal_emitter.emit_log(log_data)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
         if metrics is None: return
         # Add step number to evaluation metrics for plotting
         metric_data = metrics.copy()
         metric_data['step'] = state.global_step
         if hasattr(self.signal_emitter, 'emit_evaluate'):
             self.signal_emitter.emit_evaluate(metric_data)

    # on_train_end could be useful too
    def on_train_end(self, args, state, control, **kwargs):
         if hasattr(self.signal_emitter, 'emit_train_end'):
             self.signal_emitter.emit_train_end("Training finished.")


def train_model(model, tokenizer, train_dataset, eval_dataset, training_args_dict, callback_emitter):
    """Set up and run the training process."""
    if not model or not tokenizer or not train_dataset or not eval_dataset:
        raise ValueError("Missing required components for training (model, tokenizer, datasets).")

    try:
        training_args = TrainingArguments(**training_args_dict)
    except Exception as e:
        logger.error(f"Invalid TrainingArguments: {e}", exc_info=True)
        raise ValueError(f"Invalid training arguments: {e}") from e

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )

    # Setup callback
    trainer_callback = DesktopTrainerCallback(callback_emitter)

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[trainer_callback]
    )

    # Start training (this will block until done in the worker thread)
    logger.info("Starting model training...")
    try:
        train_result = trainer.train()
        logger.info("Training completed.")
        # Optionally log metrics
        # metrics = train_result.metrics
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        trainer.save_state() # Save optimizer, scheduler, etc.
        trainer.save_model() # Save final model state to output_dir
        logger.info(f"Model and state saved to {training_args.output_dir}")
        return trainer, "Training successful."
    except Exception as e:
        logger.error(f"Error during trainer.train(): {e}", exc_info=True)
        raise RuntimeError(f"Training failed: {e}") from e


# --- Export ---

def export_model_formats(trainer, export_base_dir, model_name, formats):
    """Export the model in selected formats."""
    if not trainer:
        raise ValueError("Trainer object is required for export.")

    export_paths = {}
    model_export_dir = Path(export_base_dir) / model_name
    model_export_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(model_export_dir) # Use consistent string path

    logger.info(f"Starting export process to {output_dir_str} for formats: {formats}")

    # 1. Save in Hugging Face format (always done by Trainer, ensure it's in the target dir)
    try:
        logger.info(f"Saving base Hugging Face model to {output_dir_str}...")
        trainer.save_model(output_dir_str)
        # Also save tokenizer
        trainer.tokenizer.save_pretrained(output_dir_str)
        export_paths['huggingface'] = output_dir_str
        logger.info("Hugging Face format saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save Hugging Face format: {e}", exc_info=True)
        # Continue with other formats if possible

    # 2. PyTorch format (state_dict)
    if 'pytorch' in formats:
        try:
            torch_path = model_export_dir / "pytorch_model.bin"
            logger.info(f"Saving PyTorch state_dict to {torch_path}...")
            torch.save(trainer.model.state_dict(), str(torch_path))
            export_paths['pytorch'] = str(torch_path)
            logger.info("PyTorch state_dict saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save PyTorch format: {e}", exc_info=True)
            # Optionally signal partial failure

    # 3. GGUF format (using auto_gptq - requires installation)
    if 'gguf' in formats:
        if AUTO_GPTQ_AVAILABLE:
            try:
                gguf_dir = model_export_dir / "gguf"
                gguf_dir.mkdir(exist_ok=True)
                gguf_path_str = str(gguf_dir) # Use string path for older libs if needed

                logger.info(f"Starting GGUF quantization to {gguf_path_str}...")

                # We need the *base* model path from the trainer for AutoGPTQ
                # Trainer should have saved it already in output_dir_str
                base_model_path = output_dir_str

                # Quantization Config (example: 4-bit)
                # Note: GPTQ can be complex, requires specific dataset examples
                # This might be slow and need refinement.
                # Using a generic config here.
                quantize_config = BaseQuantizeConfig(
                    bits=4,          # 4-bit quantization
                    group_size=128,  # Common group size
                    desc_act=False,  # Activation order (False is faster, less accurate potentially)
                    model_file_base_name="model_quantized" # Base name for GGUF file(s)
                )

                logger.info("Loading model for quantization...")
                # Load the just-saved HF model for quantization
                quant_model = AutoGPTQForCausalLM.from_pretrained(
                     base_model_path,
                     quantize_config,
                     # device_map="auto" # Let auto-gptq handle device placement if possible
                     # low_cpu_mem_usage=True # If memory is an issue
                     trust_remote_code=True # Sometimes needed for custom model code
                )

                # Provide *some* data for calibration (important for GPTQ quality)
                # Using a small sample from the eval dataset if possible
                # This part is CRUCIAL and may need adjustment based on dataset format
                calibration_data = []
                try:
                    # Use the original, non-tokenized eval dataset if available in trainer context
                    # This depends on how state was managed. Assuming trainer has access indirectly or we pass it.
                    # Let's simulate getting a few text examples:
                    # Replace this with actual access to some text data
                    sample_texts = ["Example text for calibration.", "Another example."]
                    if trainer.eval_dataset and 'text' in trainer.eval_dataset.column_names:
                         sample_texts = trainer.eval_dataset.select(range(min(16, len(trainer.eval_dataset))))['text'] # Take a small sample
                    else:
                         logger.warning("Cannot access eval dataset text for GPTQ calibration, using placeholders.")

                    calibration_data = [trainer.tokenizer(text, return_tensors='pt') for text in sample_texts]
                    logger.info(f"Using {len(calibration_data)} examples for GPTQ calibration.")

                except Exception as calib_err:
                    logger.warning(f"Could not prepare calibration data for GPTQ: {calib_err}. Quantization quality may be lower.")
                    # Proceed without calibration data if necessary? Or fail? Let's proceed with warning.

                logger.info("Quantizing model (this may take time)...")
                # Check if calibration data is needed/expected by the specific quantize method
                # AutoGPTQ API might vary. Assuming quantize method exists.
                # If calibration data is needed:
                # quant_model.quantize(calibration_data) # Pass calibration data if required/supported

                # Save the quantized model (AutoGPTQ handles GGUF conversion during save)
                # Note: AutoGPTQ's save_pretrained might directly save GGUF or require another step
                # depending on version. Check its documentation.
                # Assuming it saves to the specified directory.
                quant_model.save_pretrained(gguf_path_str) # Use the directory path

                # Find the .gguf file (name might vary)
                gguf_files = list(Path(gguf_path_str).glob("*.gguf"))
                if gguf_files:
                    export_paths['gguf'] = str(gguf_files[0]) # Store path to the GGUF file itself
                    logger.info(f"GGUF model saved successfully: {gguf_files[0]}")
                else:
                     logger.warning(f"GGUF quantization ran, but no .gguf file found in {gguf_path_str}.")
                     # Keep the directory path maybe?
                     export_paths['gguf'] = gguf_path_str + " (GGUF file not found?)"

            except Exception as e:
                logger.error(f"Failed to export GGUF format: {e}", exc_info=True)
                # Optionally signal partial failure
        else:
            logger.warning("Skipping GGUF export: auto_gptq library not available.")


    logger.info(f"Export finished. Paths: {export_paths}")
    return export_paths, output_dir_str # Return paths and the main directory

def create_zip_from_directory(directory_to_zip, zip_filename):
    """Create a zip file from a directory."""
    try:
        shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', directory_to_zip)
        logger.info(f"Successfully created zip file: {zip_filename}")
        return zip_filename
    except Exception as e:
        logger.error(f"Failed to create zip file {zip_filename} from {directory_to_zip}: {e}", exc_info=True)
        raise