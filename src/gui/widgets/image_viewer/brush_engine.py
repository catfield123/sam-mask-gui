"""Brush drawing engine for painting/erasing on segmentation masks."""

from typing import Optional, Tuple

import cv2
import numpy as np


class BrushEngine:
    """Manages brush state and performs mask painting operations.

    The engine keeps track of the current stroke: once ``start_stroke``
    is called, subsequent ``continue_stroke`` calls draw into the
    internal ``brush_mask``.  ``finalize_stroke`` returns the completed
    mask and resets the state.
    """

    def __init__(self):
        """Initialise the brush engine with no active stroke."""
        self.is_drawing: bool = False
        self.brush_mask: Optional[np.ndarray] = None
        self.brush_mode: Optional[int] = None  # 255 = add, 0 = erase
        self.last_pos: Optional[Tuple[int, int]] = None

    def start_stroke(
        self,
        x: int,
        y: int,
        mode: int,
        current_mask: Optional[np.ndarray],
        img_shape: Tuple[int, int],
        radius: int,
    ) -> np.ndarray:
        """Begin a new brush stroke.

        Args:
            - x (int): Starting X in image pixel space.
            - y (int): Starting Y in image pixel space.
            - mode (int): Paint value (255 = add foreground, 0 = erase).
            - current_mask (np.ndarray | None): Existing mask to paint on top
              of, or ``None`` to start from a blank mask.
            - img_shape (tuple[int, int]): ``(height, width)`` of the image.
            - radius (int): Brush radius in image pixels.

        Returns:
            - np.ndarray: The mask after the initial circle is drawn.
        """
        self.is_drawing = True
        self.brush_mode = mode
        self.last_pos = (x, y)

        h, w = img_shape
        if current_mask is not None and current_mask.shape[:2] == (h, w):
            self.brush_mask = current_mask.copy()
        else:
            self.brush_mask = np.zeros((h, w), dtype=np.uint8)

        self._draw_circle(self.brush_mask, (x, y), radius, mode)
        return self.brush_mask

    def continue_stroke(self, x: int, y: int, radius: int) -> Optional[np.ndarray]:
        """Continue the current stroke to a new position.

        Draws a circle at ``(x, y)`` and a connecting line from the
        previous position for smooth coverage.

        Args:
            - x (int): Current X in image pixel space.
            - y (int): Current Y in image pixel space.
            - radius (int): Brush radius in image pixels.

        Returns:
            - np.ndarray | None: Updated mask, or ``None`` if no stroke is active.
        """
        if not self.is_drawing or self.brush_mask is None or self.brush_mode is None:
            return None

        self._draw_circle(self.brush_mask, (x, y), radius, self.brush_mode)

        if self.last_pos is not None:
            self._draw_line(self.brush_mask, self.last_pos, (x, y), radius * 2, self.brush_mode)

        self.last_pos = (x, y)
        return self.brush_mask

    def finalize_stroke(self) -> Optional[np.ndarray]:
        """End the current stroke and return the final mask.

        Returns:
            - np.ndarray | None: Completed mask copy, or ``None`` if no stroke
              was active.
        """
        if not self.is_drawing or self.brush_mask is None:
            return None

        result = self.brush_mask.copy()
        self.is_drawing = False
        self.last_pos = None
        self.brush_mode = None
        return result

    def cancel(self):
        """Abort the current stroke without returning a result."""
        self.is_drawing = False
        self.last_pos = None
        self.brush_mode = None
        self.brush_mask = None

    # ------------------------------------------------------------------
    # Private drawing primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_circle(mask: np.ndarray, center: Tuple[int, int], radius: int, value: int):
        """Draw a filled circle on *mask*.

        Args:
            - mask (np.ndarray): Target mask array (modified in-place).
            - center (tuple[int, int]): Circle centre ``(x, y)``.
            - radius (int): Circle radius in pixels.
            - value (int): Fill value (0 or 255).
        """
        cv2.circle(mask, center, radius, int(value), -1)

    @staticmethod
    def _draw_line(
        mask: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        thickness: int,
        value: int,
    ):
        """Draw a line segment on *mask*.

        Args:
            - mask (np.ndarray): Target mask array (modified in-place).
            - start (tuple[int, int]): Start point ``(x, y)``.
            - end (tuple[int, int]): End point ``(x, y)``.
            - thickness (int): Line thickness in pixels.
            - value (int): Fill value (0 or 255).
        """
        cv2.line(mask, start, end, int(value), thickness)
