import argparse
import os
import cv2
from morphing import *

def parse_args():
    parser = argparse.ArgumentParser(description='Morphing')
    parser.add_argument('--source', type=str, required=True, help='Path to the source image')
    parser.add_argument('--target', type=str, required=True, help='Path to the target image')
    parser.add_argument('--output', type=str, default= "./Output", help='Path to the output directory')
    return parser.parse_args()



def LandmarkTest(image):
    detector = LandmarkDetector()
    detection_result = detector.detect_landmarks(image)
    LandmarkDetector.show_annotation(image, detection_result)

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    source = cv2.cvtColor(cv2.imread(args.source, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(cv2.imread(args.target, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    morpher = Morpher(source, target)
    
    # LandmarkTest(source)
    # morpher.show_triangles(0.5)
    
    # image = morpher.morph(0.5)
    # cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    
    for alpha in np.arange(0.0, 1.1, 0.1):
        image = morpher.morph(alpha)
        cv2.imwrite(os.path.join(args.output, f"morphed_{alpha:.1f}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

