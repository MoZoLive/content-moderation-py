from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
import keras

app = Flask(__name__)
def load_images(image):
    image = tf.keras.utils.img_to_array(image)
    image /= 255
    image = np.expand_dims(image, axis=0)
    return np.asarray(image)

class keras_predictor():
    nsfw_model = None

    def __init__(self, model_path):
        keras_predictor.nsfw_model = keras.models.load_model(model_path)

    def predict(self, image_path, batch_size=32, categories=['drawings_simple_cartoons', 'hentai_sexy_porn_cartoons', 'neutral_simple_human', 'porn_private_parts', 'sexy_nude']):
        #calling load_images method
        loaded_image = load_images(image_path)
        model_preds = keras_predictor.nsfw_model.predict(loaded_image, batch_size=batch_size)
        preds = np.argsort(model_preds, axis=1).tolist()
        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(str(round(model_preds[i][pred],2)))
                preds[i][j] = categories[pred]
            probs.append(single_probs)

        image_preds = {}
        for i in range(len(preds)):
            for _ in range(len(preds[i])):
                image_preds[preds[i][_]] = probs[i][_]
        return image_preds

weights_path = 'Nudity-Detection-Model-zipped/model.h5'
object = keras_predictor(weights_path)


@app.route("/test", methods=["GET"])
def test():
    return "<h1>This API works from your emulator/phone!</h1>"
@app.route('/detect', methods=['POST'])

def detect():
    file = base64.decodebytes(request.json["image"].encode())
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_UNCHANGED)
    image=cv2.resize(image, (299,299))
    # process image
    dct=object.predict(image)
    if not dct:
        return jsonify({"prediction":"null"})
    return jsonify(list(max(dct.items(), key = lambda x: x[1])))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
