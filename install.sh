#!/bin/bash

# MixVLLM Automated Installation Script
# This script automates the manual installation steps from linux_installation_guide.md
# Run as a non-root user with sudo privileges. Tested on Ubuntu 22.04+.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
    exit 1
fi

# Check Ubuntu version
if ! grep -q "Ubuntu" /etc/os-release; then
    print_warning "This script is designed for Ubuntu. It may not work on other distributions."
fi

UBUNTU_VERSION=$(lsb_release -rs | cut -d'.' -f1)
if [[ $UBUNTU_VERSION -lt 22 ]]; then
    print_warning "Ubuntu 22.04+ is recommended. Current version: $(lsb_release -d | cut -d':' -f2 | xargs)"
fi

# Step 1: Install base tools
print_status "Step 1: Installing base tools..."
sudo apt update
sudo apt install -y git python3 python3-venv curl

# Install uv
print_status "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    print_error "uv installation failed. Please check your PATH."
    exit 1
fi

print_status "uv version: $(uv --version)"

# Step 2: Create application directory
INSTALL_DIR="$HOME/bin/mixvllm"
print_status "Step 2: Creating application directory at $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Step 3: Initialize uv environment
print_status "Step 3: Initializing uv environment..."
uv init --no-readme --no-workspace
uv add "git+https://github.com/geosp/mixvllm.git"
uv sync

# Step 4: Add launch scripts and configs
print_status "Step 4: Downloading launch scripts and configs..."
curl -o launch https://raw.githubusercontent.com/geosp/mixvllm/main/launch
curl -o chat https://raw.githubusercontent.com/geosp/mixvllm/main/chat
chmod +x launch chat

mkdir -p configs
cd configs
CONFIG_FILES=(
    "example_model.yaml"
    "gpt-oss-20b.yaml"
    "llama-70b-tp2.yaml"
    "llama-7b.yaml"
    "mcp_servers.yaml"
    "phi3-mini.yaml"
    "phi3-mini-with-terminal.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    curl -O "https://raw.githubusercontent.com/geosp/mixvllm/main/configs/$config"
done
cd ..

# Step 5: Test the application (optional, commented out by default)
# print_status "Step 5: Testing the application..."
# uv run bash launch --model gpt-oss-20b --terminal &
# TEST_PID=$!
# sleep 10  # Wait for startup
# if curl -s http://localhost:8000/health > /dev/null; then
#     print_status "Test successful!"
# else
#     print_warning "Test failed. Please check logs."
# fi
# kill $TEST_PID 2>/dev/null || true

# Step 6: Prompt for Hugging Face token
print_status "Step 6: Hugging Face token setup..."
read -p "Do you have a Hugging Face token? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face token: " -s HF_TOKEN
    echo
    export HF_TOKEN="$HF_TOKEN"
    echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$HOME/.bashrc"
    print_status "HF_TOKEN set and added to ~/.bashrc"
else
    print_warning "Skipping HF token setup. You can set it later with: export HF_TOKEN=your_token"
fi

# Step 7: Create systemd service
print_status "Step 7: Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/mixvllm.service"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=MixVLLM Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$HOME/.local/bin/uv run bash launch --model gpt-oss-20b --terminal
Restart=on-failure
RestartSec=5s

Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin"
Environment="HOME=$HOME"
Environment="UV_CACHE_DIR=$HOME/.cache/uv"
Environment="PYTHONUNBUFFERED=1"
$(if [[ -n "$HF_TOKEN" ]]; then echo "Environment=\"HF_TOKEN=$HF_TOKEN\""; fi)

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Step 8: Enable and start the service
print_status "Step 8: Enabling and starting the service..."
sudo systemctl daemon-reload
sudo systemctl enable mixvllm.service
sudo systemctl start mixvllm.service

# Step 9: Verify the service
print_status "Step 9: Verifying the service..."
sleep 5  # Give it time to start
if sudo systemctl is-active --quiet mixvllm.service; then
    print_status "Service is running!"
    if curl -s http://localhost:8000/health | grep -q '"status":"ok"'; then
        print_status "Health check passed!"
    else
        print_warning "Health check failed. Check logs with: journalctl -u mixvllm.service -f"
    fi
else
    print_error "Service failed to start. Check status with: sudo systemctl status mixvllm.service"
    exit 1
fi

print_status "Installation complete!"
print_status "To view logs: journalctl -u mixvllm.service -f"
print_status "To manage service: sudo systemctl {start|stop|restart} mixvllm.service"
print_status "Default model: gpt-oss-20b. Edit $SERVICE_FILE to change model."