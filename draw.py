import pygame
import numpy as np
from constants import *

def draw_face_mesh(surface, points):
    # Draw the points
    for pt in points.values():
        xy = pt.i
        z = pt.n[2] * CAPTURE_WIDTH
        color = (0, 128 - min(max(z, -127), 127), 0) # from black to light green
        pygame.draw.circle(surface, color, xy, 1)

def draw_orientation(surface, orientation, origin):
    pygame.draw.line(surface, RED, origin, (origin[0] + 50*orientation[0, 0], origin[1] + 50*orientation[1, 0]))
    pygame.draw.line(surface, GREEN, origin, (origin[0] + 50*orientation[0, 1], origin[1] + 50*orientation[1, 1]))
    pygame.draw.line(surface, BLUE, origin, (origin[0] + 50*orientation[0, 2], origin[1] + 50*orientation[1, 2]))

def draw_projected_face(surface, points, origin):
    SCALE = 30

    # Draw the points
    for (i, pt) in points.items():
        color = GREEN if i < 1000 else RED
        pygame.draw.circle(surface, color, origin - SCALE*pt.p, 1)

    # Draw some lines
    for line in [
        LEFT_EYE_UPPER_LINE, LEFT_EYE_LOWER_LINE, RIGHT_EYE_UPPER_LINE, RIGHT_EYE_LOWER_LINE,
        MOUTH_UPPER_LINE, MOUTH_LOWER_LINE, LEFT_EYEBROW_LINE, RIGHT_EYEBROW_LINE
    ]:
        pygame.draw.lines(surface, RED, False, [origin - SCALE*points[i].p for i in line])

def draw_squareface(surface, mesh, orientation, origin):
    SCALE = 60

    # Face
    for line in mesh:
        transformed_points = [
            orientation[:2, :] @ np.array((SCALE*pt[0], SCALE*pt[1], -SCALE)) + origin for pt in line.points
        ]
        pygame.draw.lines(surface, line.color, line.closed, transformed_points)
