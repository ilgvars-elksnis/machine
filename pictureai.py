from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()

# prediction algorithms can be changed by changing TypeAs
prediction.setModelTypeAsInceptionV3()
# then you need to adjust path file to chosen algorithm
prediction.setModelPath(os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "car.webp"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)