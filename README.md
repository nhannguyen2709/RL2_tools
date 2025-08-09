# RL2 Tools


## Install
```
# Clone your repo (includes submodules)
git clone --recursive https://github.com/nhannguyen2709/RL2_tools.git

# If you forgot --recursive
git submodule update --init --recursive

# Install RL2 in development mode
cd RL2_tools/RL2
pip install -e .
cd ..

# Install your additional tools
pip install -r requirements.txt
```

## Update submodule
```
# Check current RL2 version
cd RL2
git log --oneline -1

# Update to latest (or specific commit)
git fetch origin
git checkout main  # or specific commit hash
cd ..

# Commit the submodule update
git add RL2
git commit -m "Update RL2 to latest/commit-hash"
git push
```