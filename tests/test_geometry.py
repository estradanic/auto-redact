import math

from auto_redact.geometry import (
    box_iou,
    clamp01,
    clamped_area,
    lerp_box,
    overlap_fraction,
    to_pixel_box,
)


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert math.isclose(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)


def test_iou_with_an_empty_box_is_zero():
    assert box_iou([5, 5, 5, 5], [0, 0, 10, 10]) == 0.0


def test_lerp_box():
    assert lerp_box([0, 0, 10, 10], [10, 10, 20, 20], 0.5) == [5, 5, 15, 15]


def test_clamped_area_only_counts_the_visible_part():
    assert clamped_area([-5, -5, 5, 5], 100, 100) == 25
    assert clamped_area([90, 90, 120, 120], 100, 100) == 100
    assert clamped_area([200, 200, 300, 300], 100, 100) == 0


def test_overlap_fraction():
    assert overlap_fraction([0, 0, 10, 10], [5, 0, 20, 10], 100, 100) == 0.5
    assert overlap_fraction([0, 0, 10, 10], [50, 50, 60, 60], 100, 100) == 0.0


def test_clamp01_handles_non_finite_values():
    assert clamp01(1.5) == 1.0 and clamp01(-1) == 0.0
    assert clamp01(float("nan")) == 0.0 and clamp01(float("inf")) == 0.0


def test_to_pixel_box_rounds_outward_and_clamps():
    assert to_pixel_box([1.2, 2.2, 3.1, 4.9], 10, 10) == (1, 2, 4, 5)
    assert to_pixel_box([-3, -3, 50, 50], 10, 10) == (0, 0, 10, 10)
    assert to_pixel_box([20, 20, 30, 30], 10, 10) is None
