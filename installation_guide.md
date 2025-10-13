# MixVLLM Installation Guide

This guide explains how to install MixVLLM using the automated `install.sh` script. This script simplifies the manual setup process into a single command, handling dependencies, environment setup, and systemd service configuration.

## Prerequisites

- **Operating System**: Ubuntu 22.04 or later (recommended).
- **Hardware**: A machine with sufficient resources for your chosen model (e.g., GPU for CUDA-based models).
- **User Account**: A non-root user with `sudo` privileges.
- **Internet Access**: Required for downloading dependencies and files.
- **Permissions**: Ensure your user can run `sudo` commands without a password prompt (or be prepared to enter it during installation).

## Quick Installation

1. **Download the Installation Script**:
   ```bash
   curl -O https://raw.githubusercontent.com/geosp/mixvllm/main/install.sh
   ```

2. **Make the Script Executable**:
   ```bash
   chmod +x install.sh
   ```

3. **Run the Installation Script**:
   ```bash
   ./install.sh
   ```

4. **Follow Prompts**:
   - The script will check your system and install dependencies.
   - When prompted, provide your Hugging Face token if you have one (optional, but required for some models).
   - The script will set up the environment, download configs, and start the service.

5. **Verify Installation**:
   - Check service status: `sudo systemctl status mixvllm.service`
   - Test the API: `curl http://localhost:8000/health` (should return `{"status":"ok"}`)

## What the Script Does

The `install.sh` script automates the following steps:

- **System Checks**: Verifies Ubuntu version and ensures it's not run as root.
- **Dependency Installation**: Updates apt and installs `git`, `python3`, `python3-venv`, `curl`, and `uv` (a fast Python package manager).
- **Environment Setup**: Creates `~/bin/mixvllm`, initializes a uv project, adds MixVLLM from GitHub, and syncs dependencies.
- **File Downloads**: Fetches `launch` and `chat` scripts, plus configuration files from the repository.
- **Configuration**: Prompts for and sets up the Hugging Face token (stored in `~/.bashrc`).
- **Service Setup**: Creates a systemd service file, enables it, and starts the service with the default model (`gpt-oss-20b`).
- **Verification**: Checks service health and API responsiveness.

## Post-Installation Configuration

### Changing the Model
The script defaults to `gpt-oss-20b`. To use a different model:

1. Edit the service file:
   ```bash
   sudo vi /etc/systemd/system/mixvllm.service
   ```

2. Change the `ExecStart` line (replace `--model gpt-oss-20b` with your desired model, e.g., `--model llama-7b`).

3. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart mixvllm.service
   ```

Available models are listed in `~/bin/mixvllm/configs/`.

### Managing the Service
- **Start**: `sudo systemctl start mixvllm.service`
- **Stop**: `sudo systemctl stop mixvllm.service`
- **Restart**: `sudo systemctl restart mixvllm.service`
- **View Logs**: `journalctl -u mixvllm.service -f`
- **Disable on Boot**: `sudo systemctl disable mixvllm.service`

### Updating MixVLLM
To update to the latest version:
```bash
cd ~/bin/mixvllm
uv sync --upgrade
sudo systemctl restart mixvllm.service
```

## Troubleshooting

### Common Issues
- **Permission Denied**: Ensure you're running as a non-root user with sudo access.
- **Service Fails to Start**: Check logs with `journalctl -u mixvllm.service -f`. Common causes: missing dependencies, invalid model, or network issues.
- **Health Check Fails**: Verify the model is loaded correctly. Try restarting the service.
- **CUDA/GPU Issues**: Ensure NVIDIA drivers and CUDA are installed if using GPU models.
- **uv Not Found**: The script adds uv to PATH, but you may need to restart your shell or run `source ~/.bashrc`.

### Manual Fallback
If the script fails, refer to `linux_installation_guide.md` for manual installation steps.

### Getting Help
- Check the repository's issues: https://github.com/geosp/mixvllm/issues
- Ensure your system meets the requirements (Python 3.10+, Ubuntu 22.04+).

## Security Notes
- The service runs as your user account (not root).
- Hugging Face tokens are stored in environment variables; keep your system secure.
- Review downloaded files for integrity if concerned about network security.

---

**MixVLLM** is now ready to use! Access the API at `http://localhost:8000` or use the provided `chat` script.