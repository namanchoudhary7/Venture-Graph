import requests, json, sys

def test_streaming_api():
    print("====================================")
    print("INITIATING PHASE 4 API VERIFICATION")
    print("====================================\n")

    url = "http://127.0.0.1:8000/api/evaluate"

    payload = {
        "idea": "An AI tool that strictly writes API integration code for Stripe and PayPal."
    }

    print(f"Connecting to {url}...")

    try:
        # Use stream=True to hold the connection open and read chunks as they arrive
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"[X] Server returned HTTP {response.status_code}")
                return
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        # Parse the SSE payload
                        event_data = json.loads(decoded_line[6:])
                        
                        if event_data["type"] == "status":
                            print(f"[STREAM] -> {event_data['message']}")
                        elif event_data["type"] == "result":
                            print("\n[STREAM] -> FINAL PAYLOAD RECEIVED:")
                            print(json.dumps(event_data['data'], indent=2))

    except requests.exceptions.ConnectionError:
        print("[X] Connection Error: Is the FastAPI server running?")

if __name__ == "__main__":
    test_streaming_api()