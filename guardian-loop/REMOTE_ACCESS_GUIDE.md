# Remote Access Guide for Guardian-Loop MI Dashboard

## 🌐 Accessing the Dashboard from Your Local Computer

You have three options for viewing the MI visualizations:

### Option 1: Streamlit Dashboard (Most Feature-Rich)

1. **SSH Tunnel Method** (Recommended):
   ```bash
   # On your local computer
   ssh -L 8501:localhost:8501 username@remote-server
   
   # Then on the remote server
   cd guardian-loop
   streamlit run mi_dashboard.py
   ```
   Access at: http://localhost:8501

2. **VS Code Remote SSH**:
   - Install "Remote - SSH" extension in VS Code
   - Connect to your remote server
   - Open terminal and run: `streamlit run mi_dashboard.py`
   - VS Code will auto-detect port 8501 and offer to forward it
   - Click "Open in Browser" when prompted

### Option 2: Gradio Dashboard (Easiest to Share)

1. **With Public URL** (No SSH needed):
   ```bash
   # On remote server - edit mi_dashboard_gradio.py
   # Change share=False to share=True in demo.launch()
   python mi_dashboard_gradio.py
   ```
   Gradio will provide a public URL like: https://xxxxx.gradio.live

2. **SSH Tunnel Method**:
   ```bash
   # On your local computer
   ssh -L 7860:localhost:7860 username@remote-server
   
   # Then on remote server
   cd guardian-loop
   python mi_dashboard_gradio.py
   ```
   Access at: http://localhost:7860

### Option 3: Static HTML Report (No Server Required)

1. **Generate the report**:
   ```bash
   # On remote server
   cd guardian-loop
   python create_mi_report.py
   ```

2. **Download and view locally**:
   ```bash
   # On your local computer
   scp username@remote-server:~/path/to/guardian-loop/mi_report.html .
   # Then open mi_report.html in your browser
   ```

3. **Or serve it temporarily**:
   ```bash
   # On remote server
   cd guardian-loop
   python -m http.server 8000
   
   # On local computer
   ssh -L 8000:localhost:8000 username@remote-server
   ```
   Access at: http://localhost:8000/mi_report.html

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Find process using port (e.g., 8501)
lsof -i :8501
# Kill the process
kill -9 <PID>
```

### Firewall Issues
If direct access doesn't work, always use SSH tunneling as it bypasses firewall restrictions.

### Multiple SSH Sessions
You can run the SSH tunnel in the background:
```bash
ssh -fN -L 8501:localhost:8501 username@remote-server
```

## 🚀 Quick Start Commands

```bash
# 1. Connect to remote server with port forwarding
ssh -L 8501:localhost:8501 -L 7860:localhost:7860 username@remote-server

# 2. Navigate to project
cd ~/path/to/guardian-loop

# 3. Run your preferred dashboard
# Option A: Streamlit
streamlit run mi_dashboard.py

# Option B: Gradio  
python mi_dashboard_gradio.py

# Option C: Static report
python create_mi_report.py
```

## 📱 Mobile Access

For mobile access, use Gradio with `share=True` or set up a reverse proxy with ngrok:
```bash
# Install ngrok
pip install pyngrok

# Run in Python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

## 🔒 Security Notes

- SSH tunneling is the most secure method
- Avoid exposing ports directly on public servers
- Use strong passwords and SSH keys
- Consider using a VPN for additional security

---

**Pro Tip**: VS Code's Remote SSH extension provides the best development experience with automatic port forwarding and integrated terminal. 