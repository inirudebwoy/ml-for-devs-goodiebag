# What is it

Example code for the presentation "ML for develpers".
Azure managed online endpoint serving image recognition
[model](https://huggingface.co/google/vit-base-patch16-224).
The code will spin up local endpoint. Cloud endpoint needs a beefy machine and thus subscription is required.

# How to use it

```
cd azure_online_endpoint
pipenv install
pipenv run python azure_inference.py
```

# How to call it

Azure endpoint requires certain pre processing of the image. The code below explains how to do it.

```python

import base64

# open jpg file and encode it as base64
with open("./000000039769.jpg", "rb") as f:
    image = f.read()

image = base64.b64encode(image).decode("ascii")

# save image in json format
with open("./image-request.json", "w") as f:
    f.write(f'{{"data": "{image}"}}')

```

```
# figure out port number by looking at ports returned
docker ps

# httpie

http post :32773/score @image-request.json

# curl

curl --request POST "http://localhost:32773/score" --header 'Content-Type: application/json' --data @image-request.json
```

In both cases you should get "Egyptian cat"

