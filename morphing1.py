import cv2
import numpy as np
import scipy
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import copy

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class LandmarkDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path='face_landmarker.task')
        self.options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                    output_face_blendshapes=True,
                                                    output_facial_transformation_matrixes=True,
                                                    num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def detect_landmarks(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)
        return detection_result


class Triangle:
    def __init__(self, vertices):
        if isinstance(vertices, np.ndarray) == 0:
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError(
                "Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")
        self.vertices = vertices

    # Credit to https://github.com/zhifeichen097/Image-Morphing for the following approach (which is a bit more efficient than my own)!
    def getPoints(self):
        width = round(max(self.vertices[:, 0]) + 2)
        height = round(max(self.vertices[:, 1]) + 2)
        mask = Image.new('P', (width, height), 0)
        ImageDraw.Draw(mask).polygon(
            tuple(map(tuple, self.vertices)), outline=255, fill=255)
        coordArray = np.transpose(np.nonzero(mask))

        return coordArray


class Morpher:
    def __init__(self, source, target):
        self.source = source.astype(np.uint8)
        self.target = target.astype(np.uint8)
        self.detector = LandmarkDetector()
        self.crop_size()
        self.source_landmarks = self.detector.detect_landmarks(source)
        self.target_landmarks = self.detector.detect_landmarks(target)
        self.source_points = self.get_points(
            self.source, self.source_landmarks)
        self.target_points = self.get_points(
            self.source, self.target_landmarks)
        print(self.source_points.shape)
        print(self.target_points.shape)
        self.source_triangle, self.target_triangle = self.loadTriangles()
        self.new_source_image = copy.deepcopy(self.source)
        self.new_target_image = copy.deepcopy(self.target)

    def crop_size(self):
        max_height = max(self.source.shape[0], self.target.shape[0])
        max_width = max(self.source.shape[1], self.target.shape[1])

        width = self.source.shape[1]
        height = self.source.shape[0]
        start_x = (width - max_width) // 2
        start_y = (height - max_height) // 2
        self.source = self.source[start_y:start_y +
                                  max_height, start_x:start_x+max_width]

        width = self.target.shape[1]
        height = self.target.shape[0]
        start_x = (width - max_width) // 2
        start_y = (height - max_height) // 2
        self.target = self.target[start_y:start_y +
                                  max_height, start_x:start_x+max_width]

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

    def loadTriangles(self):
        leftTriList = []
        rightTriList = []

        delaunayTri = Delaunay(self.source_points)

        leftNP = self.source_points[delaunayTri.simplices]
        rightNP = self.target_points[delaunayTri.simplices]

        for x, y in zip(leftNP, rightNP):
            leftTriList.append(Triangle(x))
            rightTriList.append(Triangle(y))
        return leftTriList, rightTriList

    def getImageAtAlpha(self, alpha):
        for source_triangle, target_traingle in zip(self.source_triangle, self.target_triangle):
            self.interpolatePoints(source_triangle, target_traingle, alpha)
        return ((1 - alpha) * self.new_source_image + alpha * self.new_target_image).astype(np.uint8)

    def interpolatePoints(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(
            leftTriangle.vertices + (rightTriangle.vertices - leftTriangle.vertices) * alpha)
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array([[leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[0][0],
                                       leftTriangle.vertices[0][1], 1],
                                   [leftTriangle.vertices[1][0],
                                       leftTriangle.vertices[1][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[1][0],
                                       leftTriangle.vertices[1][1], 1],
                                   [leftTriangle.vertices[2][0],
                                       leftTriangle.vertices[2][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1]])
        tempRightMatrix = np.array([[rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[0][0],
                                        rightTriangle.vertices[0][1], 1],
                                    [rightTriangle.vertices[1][0],
                                        rightTriangle.vertices[1][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[1][0],
                                        rightTriangle.vertices[1][1], 1],
                                    [rightTriangle.vertices[2][0],
                                        rightTriangle.vertices[2][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1]])
        try:
            lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
            righth = np.linalg.solve(tempRightMatrix, targetVertices)
            leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [
                             lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
            rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [
                              righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
            leftinvH = np.linalg.inv(leftH)
            rightinvH = np.linalg.inv(rightH)
            targetPoints = targetTriangle.getPoints()

            # Credit to https://github.com/zhifeichen097/Image-Morphing for the following code block that I've adapted. Exceptional work on discovering
            # RectBivariateSpline's .ev() method! I noticed the method but didn't think much of it at the time due to the website's poor documentation..
            xp, yp = np.transpose(targetPoints)
            leftXValues = leftinvH[1, 1] * xp + \
                leftinvH[1, 0] * yp + leftinvH[1, 2]
            leftYValues = leftinvH[0, 1] * xp + \
                leftinvH[0, 0] * yp + leftinvH[0, 2]
            leftXParam = np.arange(np.amin(leftTriangle.vertices[:, 1]), np.amax(
                leftTriangle.vertices[:, 1]), 1)
            leftYParam = np.arange(np.amin(leftTriangle.vertices[:, 0]), np.amax(
                leftTriangle.vertices[:, 0]), 1)
            leftImageValues = self.leftImage[int(leftXParam[0]):int(
                leftXParam[-1] + 1), int(leftYParam[0]):int(leftYParam[-1] + 1)]

            rightXValues = rightinvH[1, 1] * xp + \
                rightinvH[1, 0] * yp + rightinvH[1, 2]
            rightYValues = rightinvH[0, 1] * xp + \
                rightinvH[0, 0] * yp + rightinvH[0, 2]
            rightXParam = np.arange(np.amin(rightTriangle.vertices[:, 1]), np.amax(
                rightTriangle.vertices[:, 1]), 1)
            rightYParam = np.arange(np.amin(rightTriangle.vertices[:, 0]), np.amax(
                rightTriangle.vertices[:, 0]), 1)
            rightImageValues = self.rightImage[int(rightXParam[0]):int(
                rightXParam[-1] + 1), int(rightYParam[0]):int(rightYParam[-1] + 1)]

            self.new_source_image[xp, yp] = RectBivariateSpline(
                leftXParam, leftYParam, leftImageValues, kx=1, ky=1).ev(leftXValues, leftYValues)
            self.new_target_image[xp, yp] = RectBivariateSpline(
                rightXParam, rightYParam, rightImageValues, kx=1, ky=1).ev(rightXValues, rightYValues)
        except:
            return
