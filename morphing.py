import cv2
import numpy as np
import scipy
from scipy.spatial import Delaunay   

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class LandmarkDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        self.options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
    def detect_landmarks(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        return detection_result
    @staticmethod
    def draw_landmarks_mediapipe(rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image
    @staticmethod
    def draw_landmarks(image, detection_result):
        face_landmark = detection_result.face_landmarks[0] # landmark of face 0
        annotated_image = np.copy(image)
        height = image.shape[0]
        width = image.shape[1]
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmark)):
            landmark = face_landmark[idx]
            annotated_image = cv2.circle(annotated_image, (int(landmark.x * width), int(landmark.y * height)), 1, (0, 0, 255), -1)
        return annotated_image
    @staticmethod
    def show_annotation(image, detection_result):
        # annotated_image = LandmarkDetector.draw_landmarks_mediapipe(image, detection_result)
        annotated_image = LandmarkDetector.draw_landmarks(image,detection_result)
        cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

class Morpher:
    def __init__(self, source, target):
        self.source = source.astype(np.uint8)
        self.target = target.astype(np.uint8)
        self.detector = LandmarkDetector()
        
        self.crop_size()
        self.source_landmarks = self.detector.detect_landmarks(source)
        self.target_landmarks = self.detector.detect_landmarks(target)
        self.source_points = self.get_points(self.source,self.source_landmarks)
        self.target_points = self.get_points(self.source,self.target_landmarks)
        self.source_triangles = self.get_triangles(self.source_points)
        self.target_triangles = self.get_triangles(self.target_points)
    
    def crop_size(self):
        max_height = max(self.source.shape[0], self.target.shape[0])
        max_width = max(self.source.shape[1], self.target.shape[1])
        
        width = self.source.shape[1]
        height = self.source.shape[0]
        start_x = (width - max_width) // 2
        start_y = (height - max_height) // 2
        self.source = self.source[start_y:start_y+max_height, start_x:start_x+max_width]
        
        width = self.target.shape[1]
        height = self.target.shape[0]
        start_x = (width - max_width) // 2
        start_y = (height - max_height) // 2
        self.target = self.target[start_y:start_y+max_height, start_x:start_x+max_width]
        
    def get_points(self, image, landmarks):
        points = []
        height = image.shape[0]
        width = image.shape[1]
        # Add landmarks
        for landmark in landmarks.face_landmarks[0]:
            points.append((landmark.x*width, landmark.y*height))
        # Add edge points
        points.append((0, 0))
        points.append((width//2, 0))
        points.append((width-1, 0))
        points.append((0, height//2))
        points.append((width-1, height//2))
        points.append((0, height-1))
        points.append((width//2, height-1))
        points.append((width-1, height-1))
        
        return np.array(points)
        
    def get_triangles(self, points):
        tri = Delaunay(points)
        return tri.simplices
    
    @staticmethod
    def draw_triangles(image, points, triangles):
        for triangle in triangles:
            pt1 = (int(points[triangle[0]][0]), int(points[triangle[0]][1]))
            pt2 = (int(points[triangle[1]][0]), int(points[triangle[1]][1]))
            pt3 = (int(points[triangle[2]][0]), int(points[triangle[2]][1]))
            cv2.line(image, pt1, pt2, (255, 255, 255), 1)
            cv2.line(image, pt2, pt3, (255, 255, 255), 1)
            cv2.line(image, pt3, pt1, (255, 255, 255), 1)
        return image
    
    def show_triangles(self):
        source_image = np.copy(self.source)
        target_image = np.copy(self.target)
        source_image = self.detector.draw_landmarks(source_image, self.source_landmarks)
        target_image = self.detector.draw_landmarks(target_image, self.target_landmarks)
        source_image = self.draw_triangles(source_image, self.source_points, self.source_triangles)
        target_image = self.draw_triangles(target_image, self.target_points, self.target_triangles)
        cv2.imshow("Source Triangles", cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Target Triangles", cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    
    def morph(self, alpha):
        alpha = np.clip(alpha, 0, 1)
        morphed_image = np.zeros(self.source.shape, dtype=np.float64)
        for source_triangle, target_triangle in zip(self.source_triangles,self.target_triangles):
            # Get the points of the triangle
            source_triangle = self.source_points[source_triangle]
            target_triangle = self.target_points[target_triangle]
            morphed_triangle = (1-alpha)*source_triangle + alpha*target_triangle
            morphed_triangle = morphed_triangle.astype(np.float32)
            # Get the bounding box of the triangle
            x, y, w, h = cv2.boundingRect(morphed_triangle)
            # Get the affine transformation matrix
            source_triangle = source_triangle.astype(np.float32)
            target_triangle = target_triangle.astype(np.float32)
            morphed_triangle = morphed_triangle.astype(np.float32)
            M = cv2.getAffineTransform(source_triangle[:3], morphed_triangle[:3])
            N = cv2.getAffineTransform(target_triangle[:3], morphed_triangle[:3])
            # Warp the triangle
            warped_from_source = cv2.warpAffine(self.source, M, morphed_image.shape[:2])
            warped_from_target = cv2.warpAffine(self.target, N, morphed_image.shape[:2])
            mask = np.zeros(morphed_image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, morphed_triangle[:3].astype(np.int32), 255)    
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            morphed_image += warped_from_source * (mask//255) * (1-alpha) + warped_from_target * (mask//255) * alpha
        return morphed_image.astype(np.uint8)