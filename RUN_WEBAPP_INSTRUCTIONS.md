# Running MI Micro Judges Web App via SSH Tunnel

## Option 1: SSH Port Forwarding (Recommended)

### Step 1: Set up SSH tunnel
Open a new terminal on your local machine and run:

```bash
ssh -i /path/to/your/key.pem -L 8501:localhost:8501 ubuntu@<your-lambda-instance-ip>
```

Or if you're using the Lambda.ai hostname:
```bash
ssh -i /path/to/your/key.pem -L 8501:localhost:8501 ubuntu@lambda-ai-tom
```

This creates a tunnel that forwards port 8501 from the remote server to your local machine.

### Step 2: Run the Streamlit app on Lambda.ai
In your Cursor terminal (already connected to Lambda.ai), run:

```bash
cd /home/ubuntu/stathis/martian-hackathon
streamlit run guardian-loop/mi_dashboard.py --server.headless true
```

### Step 3: Access in your browser
Open your local browser and go to:
```
http://localhost:8501
```

## Option 2: Gradio Dashboard (Public URL)

Gradio can create a public URL automatically:

```bash
cd /home/ubuntu/stathis/martian-hackathon
python guardian-loop/mi_dashboard_gradio.py
```

This will print a public URL like `https://xxxxx.gradio.live` that you can access from anywhere.

## Option 3: Static HTML Report

Generate a static report and view it locally:

```bash
# On Lambda.ai:
cd /home/ubuntu/stathis/martian-hackathon
python guardian-loop/create_mi_report.py

# Then download it to your local machine:
scp -i /path/to/your/key.pem ubuntu@<lambda-ip>:/home/ubuntu/stathis/martian-hackathon/mi_report.html ./
```

Then open `mi_report.html` in your local browser.

## Option 4: Simple HTTP Server

For viewing existing visualizations:

```bash
# On Lambda.ai:
cd /home/ubuntu/stathis/martian-hackathon/outputs/checkpoints/training_visualizations
python -m http.server 8888

# On your local machine (new terminal):
ssh -i /path/to/your/key.pem -L 8888:localhost:8888 ubuntu@<lambda-ip>

# Then browse to:
http://localhost:8888
```

## Troubleshooting

1. **Port already in use**: Try a different port (e.g., 8502, 8503)
2. **Connection refused**: Make sure the app is running on Lambda.ai first
3. **Streamlit network error**: Add `--server.address 0.0.0.0` to the streamlit command

## For Cursor Users

If you want to preview directly in Cursor:
1. Use the port forwarding method above
2. In Cursor, press `Cmd/Ctrl + Shift + P`
3. Type "Simple Browser: Show"
4. Enter `http://localhost:8501` 