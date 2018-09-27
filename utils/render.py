import numpy as np


def draw_square(screen, pos, pos2, color):
    x, y, x2, y2 = pos[0], pos[1], pos2[0], pos2[1]
    for i in range(x, x2):
        for j in range(y, y2):
            screen[i, j] = color


def merge_views_vertical(views, padding=0):
    return merge_views('vertical', views, padding)


def merge_views_horizontal(views, padding=0):
    return merge_views('horizontal', views, padding)


def merge_views(direction, views, add_padding=0):
    assert(direction == 'horizontal' or direction == 'vertical')

    axis = 1 if direction == 'horizontal' else 0
    other_axis = 1-axis

    padding = np.zeros((3, 2), np.int32)
    padding[axis, 1] = add_padding

    padded_views = []
    for i, view in enumerate(views):
        pv = np.pad(view, padding.tolist(), mode='constant', constant_values=0) if i < len(views) else 0

        max_view_size = max(views, key=lambda v: v.shape[other_axis]).shape[other_axis]
        curr_view_size = pv.shape[other_axis]
        if curr_view_size < max_view_size:
            missing_for_pad = max_view_size - curr_view_size
            fit_padding = np.zeros((3, 2), np.int32)
            fit_padding[other_axis] = (missing_for_pad // 2, missing_for_pad - missing_for_pad // 2)
            pv = np.pad(pv, fit_padding, mode='constant')
        padded_views.append(pv)

    return np.concatenate(padded_views, axis=axis)
