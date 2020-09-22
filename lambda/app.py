import os
import io
import json
import boto3
import base64
import numpy as np

from chalice import Chalice
from chalice import BadRequestError
from PIL import Image, ImageOps

app = Chalice(app_name="predictor")
app.debug = True


@app.route("/", methods=["POST"])
def index():
    body = app.current_request.json_body

    if "data" not in body:
        raise BadRequestError("Missing image data.")
    if "ENDPOINT_NAME" not in os.environ:
        raise BadRequestError("Missing endpoint.")

    endpoint = os.environ["ENDPOINT_NAME"]

    img_bytes = base64.b64decode(body["data"])
    img = Image.open(io.BytesIO(img_bytes))
    img_resized = img.resize((28, 28), Image.ANTIALIAS)
    img_greyscale = ImageOps.grayscale(img_resized)
    img_arr = np.array(img_greyscale)
    img_arr32 = img_arr.astype(np.float32)
    img_arr32 = img_arr32.reshape((1, 1, 28, 28))

    payload = json.dumps(img_arr32.tolist())

    runtime = boto3.client("runtime.sagemaker")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint, ContentType="application/json", Body=payload
    )

    response = response["Body"].read()
    result = json.loads(response.decode("utf-8"))
    np_result = np.asarray(result)
    prediction = np_result.argmax(axis=1)[0]

    return {
        "response": {
            "Probabilities: ": np.array_str(np_result),
            "This is your number: ": str(prediction),
        }
    }
