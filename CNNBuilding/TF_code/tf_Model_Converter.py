#To convert a saved model to a TFlite Model
import tensorflow as tf




#Converting model to a TFLite model that can be used for mobile applications
converter = tf.lite.TFLiteConverter.from_saved_model("C:\\Coding\\Github\\Project_3DPrintingApp\\CNNBuilding\\TF_code")
print(converter)
tflite_model = converter.convert()

#Saved converted model 
with open ('3dPrintModel.tflite', 'wb') as f:
    f.write(tflite_model)