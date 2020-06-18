from base import Inference


class Head_Pose_Estimation(Inference):
    def __init__(self):
        super().__init__()

    @staticmethod
    def preprocess_output(outputs, *args, **kwargs):
        return [outputs[key][0][0] for key in outputs]

    def predict(self, image):
        coords = self.prediction_helper(image, self.preprocess_output)
        return coords
