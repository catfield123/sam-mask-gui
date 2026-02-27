"""Coordinate conversion utilities for the image viewer."""

from typing import Optional, Tuple


class CoordinateMapper:
    """Converts between widget (screen) coordinates and image pixel coordinates.

    All methods are stateless; display geometry is passed in explicitly.
    """

    @staticmethod
    def widget_to_image(
        wx: float,
        wy: float,
        display_offset_x: int,
        display_offset_y: int,
        display_scale: float,
    ) -> Tuple[int, int]:
        """Convert widget coordinates to image pixel coordinates.

        Args:
            - wx (float): X position in widget space.
            - wy (float): Y position in widget space.
            - display_offset_x (int): Horizontal pixel offset of the image
              inside the widget.
            - display_offset_y (int): Vertical pixel offset.
            - display_scale (float): Combined scale factor
              (base_display_scale * zoom_factor).

        Returns:
            - tuple[int, int]: ``(img_x, img_y)`` in image pixel space.
        """
        img_x = int((wx - display_offset_x) / display_scale)
        img_y = int((wy - display_offset_y) / display_scale)
        return img_x, img_y

    @staticmethod
    def widget_to_image_clamped(
        wx: float,
        wy: float,
        display_offset_x: int,
        display_offset_y: int,
        actual_display_w: int,
        actual_display_h: int,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[int, int]]:
        """Convert widget coordinates to image coordinates, clamped to image bounds.

        Allows the cursor to be outside the displayed image rectangle —
        the result is clamped to valid pixel indices.  Returns ``None``
        only when display dimensions are zero (no image loaded).

        Args:
            - wx (float): X in widget space.
            - wy (float): Y in widget space.
            - display_offset_x (int): Image X offset inside the widget.
            - display_offset_y (int): Image Y offset inside the widget.
            - actual_display_w (int): Rendered image width in widget pixels.
            - actual_display_h (int): Rendered image height in widget pixels.
            - img_w (int): Source image width in pixels.
            - img_h (int): Source image height in pixels.

        Returns:
            - tuple[int, int] | None: ``(img_x, img_y)`` clamped to
              ``[0, img_w-1] x [0, img_h-1]``, or ``None``.
        """
        if actual_display_w == 0 or actual_display_h == 0:
            return None

        rel_x = (wx - display_offset_x) / actual_display_w
        rel_y = (wy - display_offset_y) / actual_display_h

        img_x = max(0, min(int(rel_x * img_w), img_w - 1))
        img_y = max(0, min(int(rel_y * img_h), img_h - 1))
        return img_x, img_y

    @staticmethod
    def image_to_display(
        img_x: int,
        img_y: int,
        img_w: int,
        img_h: int,
        display_offset_x: int,
        display_offset_y: int,
        actual_display_w: int,
        actual_display_h: int,
    ) -> Tuple[int, int]:
        """Convert image pixel coordinates to widget display coordinates.

        Args:
            - img_x (int): X in image space.
            - img_y (int): Y in image space.
            - img_w (int): Source image width.
            - img_h (int): Source image height.
            - display_offset_x (int): Image X offset inside the widget.
            - display_offset_y (int): Image Y offset inside the widget.
            - actual_display_w (int): Rendered image width in widget pixels.
            - actual_display_h (int): Rendered image height in widget pixels.

        Returns:
            - tuple[int, int]: ``(display_x, display_y)`` in widget space.
        """
        rel_x = img_x / img_w if img_w > 0 else 0.0
        rel_y = img_y / img_h if img_h > 0 else 0.0
        display_x = int(display_offset_x + rel_x * actual_display_w)
        display_y = int(display_offset_y + rel_y * actual_display_h)
        return display_x, display_y

    @staticmethod
    def brush_image_radius(brush_size: int, display_scale: float) -> int:
        """Compute the brush radius in image pixels from screen-pixel diameter.

        Args:
            - brush_size (int): Brush diameter in screen (widget) pixels.
            - display_scale (float): Current display scale factor.

        Returns:
            - int: Radius in image pixels (>= 1).
        """
        if display_scale == 0:
            return 1
        return max(1, int((brush_size / 2) / display_scale))
