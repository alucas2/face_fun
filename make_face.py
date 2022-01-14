import numpy as np
from constants import *

class MeshLine:
    def __init__(self, points, color, closed):
        self.points = points
        self.color = color
        self.closed = closed

def make_line_transformed(color, line, offset, scale, angle=0, closed=False):
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return MeshLine([r @ (pt * scale) + offset for pt in line], color, closed)

def make_squareface(face_variables):
    fv = face_variables
    mesh_lines = []

    # Head
    head_contour = [np.array((-1, -1)), np.array((-1, 1)), np.array((1, 1)), np.array((1, -1))]
    mesh_lines.append(make_line_transformed(
        WHITE, head_contour, np.array((0, 0)), 1, 0, True
    ))

    # Mouth
    NEUTRAL_SMILE = -0.02
    SMILE_FACTOR = 0.2
    MOUTH_WIDTH_FACTOR = 0.25
    MOUTH_HEIGHT_FACTOR = 0.3
    MOUTH_Y_FACTOR = 0.15
    MOUTH_Y_OFFSET = -0.25

    s = fv[MOUTH_SMILE] - NEUTRAL_SMILE
    # Magic function to limit the smiling and exagerate the frowning
    s = SMILE_FACTOR * max((1 - np.exp(-40*s) + np.tanh(s)), -5.0)
    h = MOUTH_WIDTH_FACTOR * fv[MOUTH_OPEN_X]**1.5
    mouth_upper_line = [
        np.array((x, s*x**2 + MOUTH_HEIGHT_FACTOR * fv[MOUTH_OPEN_Y])) for x in np.linspace(-h, h, 16)
    ]
    mouth_lower_line = [
        np.array((x, s*x**2 - MOUTH_HEIGHT_FACTOR * fv[MOUTH_OPEN_Y])) for x in np.linspace(-h, h, 16)
    ]
    mouth_center = np.array((
        0, MOUTH_Y_OFFSET + MOUTH_Y_FACTOR * fv[MOUTH_POSITION_Y]
    ))

    mesh_lines.append(make_line_transformed(
        WHITE, mouth_upper_line + list(reversed(mouth_lower_line)), mouth_center, 1, 0, True
    ))

    # Eyes
    EYE_FIXED_WIDTH = 0.25
    EYE_HEIGHT_FACTOR = 3.0
    EYE_SPACING = 0.5
    EYE_Y_OFFSET = 0.1

    w = 0.5 * (fv[LEFT_EYE_OPEN_X] + fv[RIGHT_EYE_OPEN_X])
    h = 0.5 * (fv[LEFT_EYE_OPEN_Y] + fv[RIGHT_EYE_OPEN_Y])
    eye_width = EYE_FIXED_WIDTH
    eye_height = EYE_FIXED_WIDTH * EYE_HEIGHT_FACTOR * h / w

    eye_contour = [np.array((-1, -1)), np.array((-1, 1)), np.array((1, 1)), np.array((1, -1))]
    eye_size = np.array((eye_width, eye_height))
    left_eye_center = np.array((-EYE_SPACING, EYE_Y_OFFSET))
    right_eye_center = np.array((EYE_SPACING, EYE_Y_OFFSET))
    
    mesh_lines.append(make_line_transformed(
        WHITE, eye_contour, left_eye_center, eye_size, 0, True
    ))
    mesh_lines.append(make_line_transformed(
        WHITE, eye_contour, right_eye_center, eye_size, 0, True
    ))

    # Eyebrows
    NEUTRAL_EYEBROW_RAISE = 0.59
    EYEBROW_RAISE_FACTOR = 1.0
    EYEBROW_ANGLE_FACTOR = 0.8
    EYEBROW_RAISE_OFFSET = 0.3
    EYEBROW_FIXED_WIDTH = 0.25
    EYEBROW_FIXED_HEIGHT = 0.08

    eyebrow_contour = [np.array((-1, -1)), np.array((-1, 1)), np.array((1, 1)), np.array((1, -1))]
    eyebrow_size = np.array((EYEBROW_FIXED_WIDTH, EYEBROW_FIXED_HEIGHT))
    left_eyebrow_angle = EYEBROW_ANGLE_FACTOR*(fv[LEFT_EYEBROW_RAISE] - NEUTRAL_EYEBROW_RAISE)
    left_eyebrow_center = left_eye_center + np.array(
        (0, EYEBROW_RAISE_OFFSET + EYEBROW_RAISE_FACTOR*(fv[LEFT_EYEBROW_RAISE] - NEUTRAL_EYEBROW_RAISE))
    )
    right_eyebrow_angle = -EYEBROW_ANGLE_FACTOR*(fv[RIGHT_EYEBROW_RAISE] - NEUTRAL_EYEBROW_RAISE)
    right_eyebrow_center = right_eye_center + np.array(
        (0, EYEBROW_RAISE_OFFSET + EYEBROW_RAISE_FACTOR*(fv[RIGHT_EYEBROW_RAISE] - NEUTRAL_EYEBROW_RAISE))
    )

    mesh_lines.append(make_line_transformed(
        WHITE, eyebrow_contour, left_eyebrow_center, eyebrow_size, left_eyebrow_angle, True
    ))
    mesh_lines.append(make_line_transformed(
        WHITE, eyebrow_contour, right_eyebrow_center, eyebrow_size, right_eyebrow_angle, True
    ))

    return mesh_lines