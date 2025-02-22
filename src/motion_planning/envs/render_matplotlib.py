import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray


class MotionPlanningRender:
    def __init__(self, width, state_ndim):
        self.positions = []
        self.width = width
        self.state_ndim = state_ndim
        plt.ioff()
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.markersize = 6.0 * (1000 / self.width)
        self.edge_collection = matplotlib.collections.LineCollection(
            [], colors="gray", linewidths=1
        )
        self.ax.add_collection(self.edge_collection)
        self.target_scatter = self.ax.plot([], [], "rx", markersize=self.markersize)[0]
        self.observed_scatter = self.ax.plot([], [], "y^", markersize=self.markersize)[
            0
        ]
        self.agent_scatter = self.ax.plot([], [], "bo", markersize=self.markersize)[0]
        self.ax.set_xlim(-self.width / 2, self.width / 2)
        self.ax.set_ylim(-self.width / 2, self.width / 2)
        # get background for blitting
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def reset(self):
        pass

    def render(
        self,
        targets: NDArray,
        positions: NDArray,
        reward: float,
        coverage: float,
        edge_index: NDArray,
        observed_targets: NDArray,
    ):
        """
        Renders the environment with the given parameters.

        Args:
            targets (array-like): (N, 2) The positions of the goal targets.
            positions (array-like): (N, 2) The positions of the agents.
            reward (float): The reward value.
            edge_index (array-like): The (2, M) edge index for the graph.
            observed_targets (array-like): (N, 3, 2) The observed targets.

        Returns:
            matplotlib.figure.Figure: The rendered figure.
        """
        if not isinstance(self.fig.canvas, FigureCanvasAgg):
            raise ValueError("Only agg matplotlib backend is supported.")
        # Update or create target scatter plot
        self.target_scatter.set_data(*targets.T)
        self.agent_scatter.set_data(*positions.T)
        # Update edges
        start_pos = positions[edge_index[0]]
        end_pos = positions[edge_index[1]]
        # Create segments for LineCollection
        segments = np.stack([start_pos, end_pos], axis=1)
        self.edge_collection.set_segments(segments)  # type: ignore
        # Update observed targets
        targets = (observed_targets + positions[:, np.newaxis, :]).reshape(-1, 2)
        self.observed_scatter.set_data(*targets.T)
        # Update title
        self.ax.set_title(f"Reward: {reward:.2f}, Coverage: {np.round(coverage*100)}%")
        # blit the background
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.draw()
        img = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        return img
