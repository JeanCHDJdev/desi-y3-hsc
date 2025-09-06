import numpy as np 
import matplotlib.pyplot as plt
import warnings
import logging
import os

from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, Union

def plot_settings(custom_settings: Optional[Dict[str, Any]] = None):
    """
    Set up consistent plot settings for all figures.
    
    Parameters
    ----------
    custom_settings : dict, optional
        Custom rcParams to override defaults
    """
    # reload rcParams defaults
    plt.rcdefaults()

    # apply new default settings
    default_settings = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 14,                  
        'axes.labelsize': 16,              
        'axes.titlesize': 18,              
        'xtick.labelsize': 14,             
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'text.usetex': False,           
        'xtick.direction': 'in',           
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.5,
    }
    
    if custom_settings:
        default_settings.update(custom_settings)
    
    plt.rcParams.update(default_settings)

class PlotManager:
    """
    PlotManager wraps matplotlib figure creation for consistent styling,
    saving, and management across all plots in the project.
    """
    
    def __init__(
            self, 
            root: str = 'figures/',
            default_figsize: Tuple[float, float] = (10, 8),
            default_dpi: int = 300,
            overwrite: bool = False,
            custom_settings: Optional[Dict[str, Any]] = None
            ):
        """
        Initialize the PlotManager.
        
        Parameters
        ----------
        root : str
            Root directory for saving figures
        default_figsize : tuple
            Default figure size (width, height)
        default_dpi : int
            Default DPI for figures
        overwrite : bool
            Default overwrite behavior for existing files
        custom_settings : dict, optional
            Custom rcParams to override defaults
        """
        self.root = Path(root)
        self.default_figsize = default_figsize
        self.default_dpi = default_dpi
        self.overwrite = overwrite
        self.custom_settings = custom_settings or {}
        
        self.root.mkdir(parents=True, exist_ok=True)

        # assert user has write permissions
        if not self.root.is_dir() or not os.access(self.root, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {self.root}")
        
        # apply plot settings
        plot_settings(custom_settings)
    
    @contextmanager
    def make_plot(
        self, 
        name: str,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        formats: Union[str, list] = 'png',
        tight_layout: bool = True,
        bbox_inches: str = 'tight',
        facecolor: str = 'white',
        show: bool = False,
        nrows: int = 1,
        ncols: int = 1,
        custom_layout: bool = False,
        add_labels: bool = False,
        label_position: str = 'upper left',
        **subplot_kwargs
        ):
        """
        Context manager for creating plots with consistent settings.
        
        Parameters
        ----------
        name : str
            Name of the plot (without extension)
        figsize : tuple, optional
            Figure size (width, height)
        dpi : int, optional
            DPI for the figure
        formats : str or list
            File format(s) to save ('png', 'pdf', 'svg', etc.)
        tight_layout : bool
            Whether to use tight_layout()
        bbox_inches : str
            bbox_inches parameter for savefig
        facecolor : str
            Face color for the figure
        show : bool
            Whether to display the plot interactively using plt.show()
        nrows, ncols : int
            Number of subplot rows and columns
        custom_layout : bool
            If True, yields only the figure for custom subplot arrangements
        add_labels : bool
            Whether to add (a), (b), (c), etc. labels to subplots
        label_position : str
            Position of the labels ('upper left', 'upper right', etc.)
        **subplot_kwargs
            Additional arguments passed to plt.subplots()
        
        Yields
        ------
        fig, ax : matplotlib Figure and Axes objects
            If custom_layout=True, yields only fig
            If nrows=ncols=1, ax is a single Axes object
            Otherwise, ax is a numpy array of Axes objects
        
        Examples
        --------
        >>> pm = PlotManager()
        >>> # Single plot
        >>> with pm.make_plot('my_plot') as (fig, ax):
        ...     ax.plot(x, y)
        ...     ax.set_xlabel('my_x_label')
        ...     ax.set_ylabel('my_y_label')
        
        >>> # Subplot grid with automatic labeling
        >>> with pm.make_plot('subplots', nrows=2, ncols=2, add_labels=True) as (fig, axes):
        ...     for i, ax in enumerate(axes.flat):
        ...         ax.plot(x, y[i])
        ...         ax.set_title(f'Plot {i+1}')
        
        >>> # Complex layout (custom subplot arrangement)
        >>> with pm.make_plot('custom_layout', custom_layout=True) as fig:
        ...     ax1 = fig.add_subplot(2, 2, 1)
        ...     ax2 = fig.add_subplot(2, 2, 2)
        ...     ax3 = fig.add_subplot(2, 1, 2)  # spans bottom row
        ...     ax1.plot(x, y1)
        ...     ax2.plot(x, y2)
        ...     ax3.plot(x, y3)
        """
        # Use defaults if not specified
        figsize = figsize or self.default_figsize
        dpi = dpi or self.default_dpi
        formats = [formats] if isinstance(formats, str) else formats
        
        # Check for existing files
        file_paths = []
        for fmt in formats:
            file_path = self.root / f'{name}.{fmt}'
            file_paths.append(file_path)
            if not self.overwrite and file_path.exists():
                raise FileExistsError(f"Figure {file_path} already exists. Set overwrite=True in PlotManager to overwrite it.")

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, facecolor=facecolor, **subplot_kwargs)
        
        # For complex layouts, remove default axes and yield only the figure
        if custom_layout:
            # Remove all default axes for custom layout
            if isinstance(ax, np.ndarray):
                for axis in ax.flat:
                    axis.remove()
            else:
                ax.remove()
            
            try:
                yield fig
                
                # Apply tight layout if requested
                if tight_layout:
                    try:
                        fig.tight_layout()
                    except Exception as e:
                        warnings.warn(f"tight_layout failed: {e}")
                
                # Save in all requested formats
                saved_paths = []
                for file_path in file_paths:
                    fig.savefig(file_path, bbox_inches=bbox_inches, dpi=dpi, 
                               facecolor=facecolor, edgecolor='none')
                    saved_paths.append(file_path)
                    logging.info(f"Saved plot: {file_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                    
            finally:
                plt.close(fig)
        else:
            # Standard behavior - yield fig and axes
            try:
                yield fig, ax
                
                if ax is not None:
                    # Add labels if requested and we have multiple subplots
                    if add_labels and (nrows > 1 or ncols > 1):
                        add_subplot_labels(ax, position=label_position)
                    
                # Apply tight layout if requested
                if tight_layout:
                    try:
                        fig.tight_layout()
                    except Exception as e:
                        warnings.warn(f"tight_layout failed: {e}")
                
                # Save in all requested formats
                saved_paths = []
                for file_path in file_paths:
                    fig.savefig(file_path, bbox_inches=bbox_inches, dpi=dpi, 
                            facecolor=facecolor, edgecolor='none')
                    saved_paths.append(file_path)
                    logging.info(f"Saved plot: {file_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                    
            finally:
                plt.close(fig)
    
    def make_subplots(self, 
                     name: str,
                     nrows: int, 
                     ncols: int,
                     figsize: Optional[Tuple[float, float]] = None,
                     add_labels: bool = True,
                     label_position: str = 'upper left',
                     **kwargs):
        """
        Backward compatibility method for creating subplot grids with automatic labeling.
        This is now just a wrapper around make_plot with add_labels=True.
        
        Parameters
        ----------
        name : str
            Name of the plot (without extension)
        nrows, ncols : int
            Number of subplot rows and columns
        figsize : tuple, optional
            Figure size (width, height). If None, auto-calculates based on grid size
        add_labels : bool
            Whether to add (a), (b), (c), etc. labels to subplots
        label_position : str
            Position of the labels ('upper left', 'upper right', etc.)
        **kwargs
            Additional arguments passed to make_plot
        
        Returns
        -------
        context manager
            Context manager that yields (fig, axes)
        
        Examples
        --------
        >>> pm = PlotManager()
        >>> with pm.make_subplots('correlation_grid', 2, 2) as (fig, axes):
        ...     for i, ax in enumerate(axes.flat):
        ...         ax.plot(x, y[i])
        ...         ax.set_title(f'Dataset {i+1}')
        """        
        return self.make_plot(
            name, nrows=nrows, ncols=ncols, 
            figsize=figsize, add_labels=add_labels, 
            label_position=label_position, **kwargs
        )
    
    def save_current_figure(self, 
                           name: str,
                           formats: Union[str, list] = 'png',
                           **kwargs):
        """
        Save the current matplotlib figure.
        
        Parameters
        ----------
        name : str
            Name of the plot (without extension)
        formats : str or list
            File format(s) to save
        **kwargs
            Additional arguments passed to savefig
        
        Returns
        -------
        list or str
            Path(s) to saved file(s)
        """
        formats = [formats] if isinstance(formats, str) else formats
        
        # Check for existing files
        file_paths = []
        for fmt in formats:
            file_path = self.root / f'{name}.{fmt}'
            file_paths.append(file_path)
            if not self.overwrite and file_path.exists():
                raise FileExistsError(f"Figure {file_path} already exists. Set overwrite=True in PlotManager to overwrite it.")
        
        # Save in all requested formats
        saved_paths = []
        for file_path in file_paths:
            plt.savefig(file_path, **kwargs)
            saved_paths.append(file_path)
            logging.info(f"Saved plot: {file_path}")
        
        return saved_paths[0] if len(saved_paths) == 1 else saved_paths

def add_subplot_labels(
        axes: np.ndarray, 
        labels: Optional[list] = None,
        position: str = 'upper left',
        **text_kwargs
        ):
    """
    Add labels (a), (b), (c), etc. to subplots.
    
    Parameters
    ----------
    axes : numpy array
        Array of matplotlib axes
    labels : list, optional
        Custom labels (defaults to alphabetical)
    position : str
        Position of the label ('upper left', 'upper right', etc.)
    **text_kwargs
        Additional arguments passed to ax.text()
    """
    axes_flat = axes.flatten()
    
    if labels is None:
        labels = [f'({chr(97 + i)})' for i in range(len(axes_flat))]
    
    position_map = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
    }
    
    xy = position_map.get(position, (0.05, 0.95))
    
    text_kw = {'transform': None, 'fontsize': 16, 'fontweight': 'bold'}
    text_kw.update(text_kwargs)
    
    for ax, label in zip(axes_flat, labels):
        if text_kw['transform'] is None:
            text_kw['transform'] = ax.transAxes
        ax.text(xy[0], xy[1], label, **text_kw)