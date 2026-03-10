import requests

url = "http://127.0.0.1:9000/predict"

payload = {
    "features": [50000, 100000, 50000, 0, 50000, 1]
}

response = requests.post(url, json=payload)
print(response.json())
