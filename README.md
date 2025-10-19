# Project Setup

## Notes

- To ensure compatibility, make sure you are using the following versions:
  - Python 3.10
  - pip<24.0

- For best developer experience, install WSL if using Windows by running the following command:

    ```bash
    wsl --install
    ```

    Reference: [Install WSL | Microsoft Learn](https://learn.microsoft.com/en-us/windows/wsl/install)

## Instructions

1. Clone the repository
    
    ```bash
    git clone https://github.com/pyvsu/modified_hifi_gan_voice_cloning.git
    ```
    
2. Create virtual environment
    
    ```bash
    python3 -m venv .venv
    ```
    
3. Activate virtual environment
    
    ```bash
    source .venv/bin/activate
    ```
    
4. Verify Python and pip are from the virtual environment
    
    ```bash
    # Should return /home/user/modified-hifi-gan-voice-cloning/.venv/bin/python
    which python
    
    # Should return /home/user/modified-hifi-gan-voice-cloning/.venv/bin/pip
    which pip
    ```
    
5.  Install dependencies
    
    ```bash
    pip install -r requirements.txt
    ```
