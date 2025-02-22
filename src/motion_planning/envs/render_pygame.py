import pygame
import numpy as np
from numpy.typing import NDArray


class MotionPlanningRenderPygame:
    BACKGROUND = (255, 255, 255)  # White
    TARGET_COLOR = (255, 0, 0)  # Red
    AGENT_COLOR = (0, 0, 255)  # Blue
    EDGE_COLOR = (128, 128, 128)  # Gray
    OBSERVED_COLOR = (255, 255, 0)  # Yellow

    def __init__(self, width, state_ndim):
        self.width = width
        self.state_ndim = state_ndim
        # Initialize pygame
        pygame.init()
        self.screen_size = 640
        self.screen = pygame.Surface((self.screen_size, self.screen_size))
        # Calculate scaling factor from world coordinates to screen coordinates
        self.scale = self.screen_size / width
        # Define sizes
        self.point_radius = max(int(3 * self.scale), 5)

    def world_to_screen(self, coords: NDArray) -> NDArray[np.int32]:
        """Convert world coordinates to screen coordinates"""
        return ((coords + self.width / 2) * self.scale).astype(np.int32)

    def reset(self):
        self.screen.fill(self.BACKGROUND)

    def render(
        self, targets, positions, reward, coverage, edge_index, observed_targets
    ):
        self.screen.fill(self.BACKGROUND)

        # Convert positions to screen coordinates
        screen_positions = self.world_to_screen(positions)
        # Draw edges
        for start, end in zip(edge_index[0], edge_index[1]):
            pygame.draw.line(
                self.screen,
                self.EDGE_COLOR,
                screen_positions[:, start],  # type: ignore
                screen_positions[:, end],  # type: ignore
                1,
            )

        # Draw targets
        screen_targets = self.world_to_screen(targets)
        for x, y in zip(*screen_targets):
            pygame.draw.circle(
                self.screen, self.TARGET_COLOR, (x, y), self.point_radius
            )

        # Draw agents
        for x, y in zip(*screen_positions):
            pygame.draw.circle(self.screen, self.AGENT_COLOR, (x, y), self.point_radius)

        # Draw observed targets
        screen_observed = self.world_to_screen(
            (observed_targets + positions.T[:, np.newaxis, :]).reshape(-1, 2).T
        )
        for x, y in zip(*screen_observed):
            pygame.draw.circle(
                self.screen, self.OBSERVED_COLOR, (x, y), self.point_radius
            )

        # Draw text
        if pygame.font.get_init():
            font = pygame.font.Font(None, 36)
            text = f"Reward: {reward:.2f}, Coverage: {np.round(coverage*100)}%"
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))

        # Convert to numpy array and add alpha channel
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
