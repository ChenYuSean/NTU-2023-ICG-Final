import argparse
import os
import cv2
from LandmarkDetector import *
from morphing import *
from morphing1 import *

def parse_args():
    parser = argparse.ArgumentParser(description='Morphing')
    parser.add_argument('source', type=str,
                        help='Path to the source image')
    parser.add_argument('target', type=str,
                        help='Path to the target image')
    parser.add_argument('--output', type=str, default="./output",
                        help='Path to the output directory')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for morphing')
    parser.add_argument('--frame', type=int, default=None,
                        help='Number of frames for morphing, overwrites alpha value if set')
    parser.add_argument('--morpher', type=str, choices=['0', '1'], default='0',
                        help='Morphing method')
    parser.add_argument('--debug', action='store_true',
                        help='Run debug mode')
    
    return parser.parse_args()


def LandmarkTest(image):
    detector = LandmarkDetector()
    detection_result = detector.detect_landmarks(image)
    LandmarkDetector.show_annotation(image, detection_result)


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    source = cv2.cvtColor(cv2.imread(
        args.source, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    target = cv2.cvtColor(cv2.imread(
        args.target, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    morpher = Morpher(source, target)
    morpher1 = Morpher1(source, target)

    if args.debug:
        LandmarkTest(source)
        morpher.show_triangles(0.5)
    
    if args.morpher == '0':
        if args.alpha and not args.frame:
            image = morpher.morph(args.alpha)
            cv2.imwrite(os.path.join(args.output, f"morphed_{args.alpha:.1f}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        if args.frame:
            for i in range(args.frame + 1):
                alpha = i / args.frame
                image = morpher.morph(alpha)
                cv2.imwrite(os.path.join(args.output, f"morphed_f{i}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
    if args.morpher == '1':
        if args.alpha and not args.frame:
            image = morpher1.getImageAtAlpha(args.alpha)
            cv2.imwrite(os.path.join(args.output, f"morphed1_{args.alpha:.1f}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if args.frame:
            for i in range(args.frame + 1):
                alpha = i / args.frame 
                image = morpher1.getImageAtAlpha(alpha)
                cv2.imwrite(os.path.join(args.output, f"morphed1_f{i}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
