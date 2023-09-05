# What is it

Example code for the presentation "ML for develpers".
FastAPI app with endpoint serving image recognition
[model](https://huggingface.co/google/vit-base-patch16-224).

# How to use it

```
cd fastapi_inference
docker compose up
```

# How to call it


```
# httpie

http -f post :8000/what img@000000039769.jpg

# curl

curl --request POST "http://localhost:8000/what" --form img=@000000039769.jpg
```

In both cases you should get "Egyptian cat"