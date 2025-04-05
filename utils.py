# utils.py
import pandas as pd
from PySide6.QtCore import (QAbstractTableModel, Qt, QModelIndex, QObject, Signal)
from PySide6.QtWidgets import (QWidget, QVBoxLayout)

# Matplotlib imports for plotting within Qt
import matplotlib
matplotlib.use('QtAgg') # Use the Qt backend for Matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Pandas Table Model ---
class PandasModel(QAbstractTableModel):
    """A model to interface a Pandas DataFrame with QTableView"""
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """Return row count."""
        if parent == QModelIndex():
            return len(self._dataframe)
        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return column count."""
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        """Return data cell"""
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            try:
                value = self._dataframe.iloc[index.row(), index.column()]
                return str(value) # Always return as string for display
            except IndexError:
                return None
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        """Return header data."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._dataframe.index[section])
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        """Set item flags"""
        # Make cells non-editable by default
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def dataframe(self):
        """Return the underlying DataFrame"""
        return self._dataframe

    def update_dataframe(self, new_dataframe: pd.DataFrame):
        """Update the model with a new DataFrame."""
        self.beginResetModel()
        self._dataframe = new_dataframe.copy() # Work with a copy
        self.endResetModel()


# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    """Matplotlib canvas widget to embed in PySide6"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        # self.fig.tight_layout() # Adjust layout

    def plot(self, x_data, y_data, title="", x_label="", y_label="", series_name=""):
        """Helper to plot simple line data."""
        self.axes.cla() # Clear previous plot
        if x_data is not None and y_data is not None and len(x_data) == len(y_data):
             self.axes.plot(x_data, y_data, marker='o', linestyle='-', label=series_name)
             if series_name: # Add legend if name provided
                 self.axes.legend()
        self.axes.set_title(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True)
        self.fig.tight_layout() # Adjust layout after plotting
        self.draw() # Redraw the canvas

    def clear_plot(self):
        """Clear the plot area."""
        self.axes.cla()
        self.draw()

# --- Signal Emitter (Passed to Callbacks) ---
class CallbackSignalEmitter(QObject):
    """Emits signals from trainer callbacks to the main UI thread."""
    train_begin = Signal(int) # max_steps
    progress_update = Signal(int, int, float) # step, max_steps, percentage
    log_received = Signal(dict) # log data from trainer
    evaluate_received = Signal(dict) # metrics data from evaluation
    train_end = Signal(str) # completion message
    error_occurred = Signal(str) # For errors within callback/training

    def emit_train_begin(self, max_steps):
        self.train_begin.emit(max_steps)

    def emit_progress(self, step, max_steps, progress):
        self.progress_update.emit(step, max_steps, progress)

    def emit_log(self, logs):
        self.log_received.emit(logs)

    def emit_evaluate(self, metrics):
        self.evaluate_received.emit(metrics)

    def emit_train_end(self, message):
         self.train_end.emit(message)

    def emit_error(self, error_message):
         self.error_occurred.emit(error_message)