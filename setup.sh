curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv

source .venv/bin/activate

uv pip install -r uv.lock

uv pip install -r guardian-loop/requirements.txt

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -m ipykernel install --user --name martian-hackathon --display-name "martian-hackathon"