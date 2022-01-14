import numpy as np
from constants import *

# Low-pass filter on the face landmarks
def landmarks_low_pass(landmarks, previous_landmarks):
    if previous_landmarks is None:
        return landmarks
    elif landmarks is None:
        return previous_landmarks
    else:
        A = 0.6
        for i in range(len(previous_landmarks)):
            previous_landmarks[i].x = A*previous_landmarks[i].x + (1-A)*landmarks[i].x
            previous_landmarks[i].y = A*previous_landmarks[i].y + (1-A)*landmarks[i].y
            previous_landmarks[i].z = A*previous_landmarks[i].z + (1-A)*landmarks[i].z
        return previous_landmarks

# Point of a face in different coordinate systems
class FacePoint:
    def __init__(self):
        # I = Image coordinates in pixels relative to the capture
        self.i = None
        # N = Normalized coordinates in which the capture has width 1 
        self.n = None
        # P = Projected coordinates in which the in-between eyes length is 1
        self.p = None

def face_point_from_landmark(landmark):
    fp = FacePoint()
    fp.i = np.array((landmark.x * CAPTURE_WIDTH, landmark.y * CAPTURE_HEIGHT))
    fp.n = np.array((landmark.x, landmark.y * CAPTURE_HEIGHT / CAPTURE_WIDTH, landmark.z))
    return fp

def face_point_from_n(n):
    fp = FacePoint()
    fp.i = np.array((n[0] * CAPTURE_WIDTH, n[1] * CAPTURE_WIDTH))
    fp.n = n
    return fp

def get_face_points(landmarks):
    if landmarks is None:
        return None

    # Grab the canonical face points (mediapipe terminology)
    points = {
        i: face_point_from_landmark(l) for (i, l) in enumerate(landmarks)
    }

    # Add my points
    points[LEFT_EYE_CENTER] = face_point_from_n(
        np.mean([points[i].n for i in LEFT_EYE_LOWER_LINE + LEFT_EYE_UPPER_LINE], axis=0)
    )
    points[RIGHT_EYE_CENTER] = face_point_from_n(
        np.mean([points[i].n for i in RIGHT_EYE_LOWER_LINE + RIGHT_EYE_UPPER_LINE], axis=0)
    )
    points[MOUTH_CENTER] = face_point_from_n(
        np.mean([points[i].n for i in MOUTH_UPPER_LINE + MOUTH_LOWER_LINE], axis=0)
    )
    return points

# Get the face rotation matrix in camera space
def get_orientation(points):
    # X vector points to the right
    face_x = points[RIGHT_EYE_CENTER].n - points[LEFT_EYE_CENTER].n
    face_x /= np.linalg.norm(face_x)

    # Z vector points backwards
    face_z = np.cross(face_x, points[TOP_VERTEX].n - points[BOTTOM_VERTEX].n)
    face_z /= np.linalg.norm(face_z)

    # Y vector points up
    face_y = np.cross(face_z, face_x)

    # Make the face rotation matrix
    face_orientation = np.zeros((3, 3))
    face_orientation[:, 0] = face_x
    face_orientation[:, 1] = face_y
    face_orientation[:, 2] = face_z
    return face_orientation

# Compute the 2d projection of the face points, as if you sticked your face in a copy machine
def compute_projected_coordinates(points):
    face_orientation = get_orientation(points)
    face_center = 0.5 * (points[RIGHT_EYE_CENTER].n + points[LEFT_EYE_CENTER].n)
    scale = np.linalg.norm(points[RIGHT_EYE_CENTER].n - points[LEFT_EYE_CENTER].n)

    for pt in points.values():
        pt.p = np.array((
            np.dot(pt.n - face_center, face_orientation[:, 0]) / scale * 2,
            np.dot(pt.n - face_center, face_orientation[:, 1]) / scale * 2
        ))

# Get some variables that describe the current expression
def get_face_variables(points):
    fv = {}

    # Mouth opening
    fv[MOUTH_OPEN_X] = np.linalg.norm(points[MOUTH_LEFT_VERTEX].p - points[MOUTH_RIGHT_VERTEX].p)
    fv[MOUTH_OPEN_Y] = np.linalg.norm(points[MOUTH_LOWER_VERTEX].p - points[MOUTH_UPPER_VERTEX].p)

    # Mouth smiliness
    mouth_horz_middle = 0.5 * (points[MOUTH_LEFT_VERTEX].p + points[MOUTH_RIGHT_VERTEX].p)
    mouth_vert_middle = 0.5 * (points[MOUTH_LOWER_VERTEX].p + points[MOUTH_UPPER_VERTEX].p)
    fv[MOUTH_SMILE] = mouth_horz_middle[1] - mouth_vert_middle[1]

    # Mouth position
    in_between_eyes = 0.5 * (points[LEFT_EYE_CENTER].p + points[RIGHT_EYE_CENTER].p)
    fv[MOUTH_POSITION_X] = points[MOUTH_CENTER].p[0] - in_between_eyes[0]
    fv[MOUTH_POSITION_Y] = points[MOUTH_CENTER].p[1] - in_between_eyes[1]

    # Left eye
    fv[LEFT_EYE_OPEN_X] = np.linalg.norm(points[LEFT_EYE_LEFT_VERTEX].p - points[LEFT_EYE_RIGHT_VERTEX].p)
    fv[LEFT_EYE_OPEN_Y] = np.linalg.norm(points[LEFT_EYE_LOWER_VERTEX].p - points[LEFT_EYE_UPPER_VERTEX].p)

    # Right eye
    fv[RIGHT_EYE_OPEN_X] = np.linalg.norm(points[RIGHT_EYE_LEFT_VERTEX].p - points[RIGHT_EYE_RIGHT_VERTEX].p)
    fv[RIGHT_EYE_OPEN_Y] = np.linalg.norm(points[RIGHT_EYE_LOWER_VERTEX].p - points[RIGHT_EYE_UPPER_VERTEX].p)

    # Left eyebrow
    fv[LEFT_EYEBROW_RAISE] = np.linalg.norm(points[LEFT_EYE_CENTER].p - points[LEFT_EYEBROW_VERTEX].p)
    fv[RIGHT_EYEBROW_RAISE] = np.linalg.norm(points[RIGHT_EYE_CENTER].p - points[RIGHT_EYEBROW_VERTEX].p)

    return fv