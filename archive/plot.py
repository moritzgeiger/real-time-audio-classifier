import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(
        self,
        win_size: int = 96,
        n_wins: int = 10,
        n_bands: int = 64,
        n_classes: int = 50,
        fig_size: tuple = (8, 8),
        msd_labels: list = None,
        blit: bool = True,
    ):
        """
        Initializes a Plotter object.

        Args:
        - win_size (int): Size of the window.
        - n_wins (int): Number of windows.
        - n_bands (int): Number of frequency bands.
        - n_classes (int): Number of classes.
        - msd_labels (list): Labels for the classes.
        - fig_size (tuple): Figure size.
        - blit (bool): Whether to use blitting for faster drawing.
        """

        # initialize plots
        self.blit = blit
        self.win_size = win_size
        self.n_wins = n_wins
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.msd_labels = msd_labels

        self.spec = np.zeros((n_bands, win_size * n_wins))
        self.act = np.zeros((n_classes, n_wins))

        self.fig = plt.figure(figsize=fig_size)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)

        self.img1 = self.ax1.imshow(
            self.spec, vmin=0, vmax=1, interpolation="None", cmap="jet", aspect="auto"
        )
        self.ax1.invert_yaxis()
        self.img2 = self.ax2.imshow(
            self.act, vmin=0, vmax=1, interpolation="None", aspect="auto"
        )

        if msd_labels is not None:
            self.ax2.set_yticks(
                np.linspace(0, len(msd_labels), len(msd_labels), endpoint=False)
            )
            self.ax2.set_yticklabels(msd_labels)
            self.ax2.set_ylim(-0.5, len(msd_labels) - 0.5)

        self.fig.canvas.draw()

        if self.blit:
            # cache the background
            self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.ax2background = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

        plt.show(block=False)

    def __call__(self, new_spec_col: np.ndarray = None, new_act_col: np.ndarray = None):
        """
        Updates the plot with new data.

        Args:
        - new_spec_col (np.array): New spectrogram column data.
        - new_act_col (np.array): New activation column data.
        """
        if new_spec_col is None:
            new_spec_col = np.random.rand(self.n_bands, self.win_size)
        if new_act_col is None:
            new_act_col = np.random.rand(self.n_classes, 1)

        self.spec = np.delete(self.spec, [k for k in range(self.win_size)], 1)
        self.act = np.delete(self.act, 0, 1)

        self.spec = np.concatenate((self.spec, new_spec_col), axis=1)
        self.act = np.concatenate((self.act, new_act_col), axis=1)

        self.img1.set_data(self.spec)
        self.img1.autoscale()
        self.img2.set_data(self.act)
        self.img2.autoscale()

        if self.blit:
            # restore background
            self.fig.canvas.restore_region(self.axbackground)
            self.fig.canvas.restore_region(self.ax2background)

            # redraw just the points
            self.ax1.draw_artist(self.img1)
            self.ax2.draw_artist(self.img2)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax1.bbox)
            self.fig.canvas.blit(self.ax2.bbox)
        else:
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()
