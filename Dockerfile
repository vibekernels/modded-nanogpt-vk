FROM --platform=linux/amd64 nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Install essentials + SSH + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo git curl wget vim tmux build-essential \
    software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# SSH setup
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && rm -f /etc/ssh/ssh_host_* \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Copy repo and install dependencies from requirements.txt
COPY . /root/modded-nanogpt-vk
# actions/checkout injects GITHUB_TOKEN into .git/config as an http extraheader.
# Strip it so the credential doesn't leak into the published image.
RUN git config --global --unset-all http.https://github.com/.extraheader 2>/dev/null; \
    cd /root/modded-nanogpt-vk && git config --unset-all http.https://github.com/.extraheader 2>/dev/null; \
    true
RUN pip install --no-cache-dir -r /root/modded-nanogpt-vk/requirements.txt

WORKDIR /root/modded-nanogpt-vk

# Startup script: configure SSH keys and start sshd, then sleep forever
RUN printf '#!/bin/bash\n\
ssh-keygen -A\n\
if [ -n "$PUBLIC_KEY" ]; then\n\
  mkdir -p /root/.ssh\n\
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys\n\
  chmod 600 /root/.ssh/authorized_keys\n\
fi\n\
# Snapshot container env vars so SSH sessions inherit them.\n\
# sshd child processes do not inherit PID 1 environment;\n\
# bash sources ~/.bashrc when invoked by sshd, so we load the\n\
# snapshot from there.\n\
printenv | grep -E "^[A-Z_][A-Z0-9_]*=" | grep -v "^PUBLIC_KEY=" | \\\n\
  awk -F = '"'"'{ val=$0; sub(/^[^=]*=/, "", val); print "export " $1 "=\\\"" val "\\\"" }'"'"' > /etc/rp_environment\n\
grep -q "source /etc/rp_environment" /root/.bashrc 2>/dev/null || \\\n\
  echo "source /etc/rp_environment" >> /root/.bashrc\n\
/usr/sbin/sshd\n\
sleep infinity\n' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
# Built via GitHub Actions -> ghcr.io/vibekernels/modded-nanogpt-vk
