import cv2
import pygame
import mediapipe
import numpy as np
from draw import *
from constants import *
from face_analysis import *
from utils import *
from make_face import *

print("Initializing pygame...")
pygame.init()
main_display = pygame.display.set_mode((CAPTURE_WIDTH, CAPTURE_HEIGHT))
plot1 = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))
plot2 = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))
plot3 = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))
time = pygame.time.Clock()
font_size = 18
font = pygame.font.SysFont("courier", font_size)
milliseconds = None
run = True

print("Opening camera...")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

print("Creating image processors...")
face_mesh_detector = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=1)
landmarks = None

print("Done.")

try:
    while run:
        # --------------------------- Events ---------------------------
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

        # --------------------------- Update ---------------------------

        # Capture and image
        _, color_capture = camera.read()
        color_capture = cv2.flip(cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB), 1)
        if color_capture.shape[1] != CAPTURE_WIDTH or color_capture.shape[0] != CAPTURE_HEIGHT:
            print("Could not set the desired capture size")

        # Draw the image
        plot1.fill(BLACK)
        plot2.fill(BLACK)
        plot3.fill(BLACK)
        pygame.surfarray.blit_array(main_display, color_capture.swapaxes(0, 1))

        # Do the face detection
        result = face_mesh_detector.process(color_capture)
        if result.multi_face_landmarks is not None:
            landmarks = landmarks_low_pass(result.multi_face_landmarks[0].landmark, landmarks)
        
        # Do the processing
        points = get_face_points(landmarks)
        if points is not None:
            # Extract orientation and projection
            orientation = get_orientation(points)
            compute_projected_coordinates(points)
            face_variables = get_face_variables(points)

            # Draw stuff
            draw_face_mesh(main_display, points)
            draw_projected_face(plot2, points, np.array((RENDER_WIDTH/2, RENDER_HEIGHT/2 - 40)))
            draw_orientation(plot1, orientation, np.array((RENDER_WIDTH/2, RENDER_HEIGHT/2)))
            mesh = make_squareface(face_variables)
            draw_squareface(plot3, mesh, orientation, np.array((RENDER_WIDTH/2, RENDER_HEIGHT/2)))
            for (i, s) in enumerate(face_variables_to_str(face_variables).split('\n')):
                main_display.blit(font.render(s, False, YELLOW), (0, (i+1)*font_size))

        # --------------------------- Swap ---------------------------
        
        # Show the frame delay in the corner
        main_display.blit(plot1, (0, CAPTURE_HEIGHT-RENDER_HEIGHT))
        main_display.blit(plot2, (CAPTURE_WIDTH-RENDER_WIDTH, CAPTURE_HEIGHT-RENDER_HEIGHT))
        main_display.blit(plot3, (CAPTURE_WIDTH-RENDER_WIDTH, CAPTURE_HEIGHT-2*RENDER_HEIGHT))
        main_display.blit(font.render(str(milliseconds) + " ms", False, YELLOW), (0, 0))
        pygame.display.flip()
        # Wait after swapping
        milliseconds = time.tick(FPS)

finally:
    camera.release()
    pygame.quit()