# ssh -p [PORT] -i ~/.ssh/mli_computa root@[IP]
# Then copy/paste this script into the remote machine

apt-get update
apt-get install git
apt-get install git-lfs
apt-get install tmux -y
apt-get install nvtop -y
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

cd ~
git clone https://github.com/dhedey/mlx-8-week4-multimodel-image-captioning
cd mlx-8-week4-multimodel-image-captioning
uv sync

# Change if you're someone else!
git config --global user.email "mli@david-edey.com"
git config --global user.name "David Edey"

# You can generate a new token at https://github.com/settings/personal-access-tokens
# => Select only this repository
# => Select Read and Write access to Contents (AKA Code)

# Launches a new tmux session (with name sweep) the name is optional!
# This session can survive even if you disconnect from SSH
# => Ctrl+B enters command mode in tmux (then release ctrl)
# ==> Ctrl+B (unclick Ctrl) then D detaches from the current tmux session
# => Discover existing sessions with tmux ls
# => Reattach to the last session with tmux a (short for attach)
# => Reattach with tmux attach -t 0
# => Scroll with Ctrl+B [ then use the arrow keys or mouse to scroll up and down. Leave scroll mode with Esc or q

tmux new -s training_session

uv run huggingface-cli login # See Discord for token

# Check GPU usage with the nvtop command

# Now run a script, e.g.
# uv run -m model.start_train --wandb
# uv run -m model.continue_train --wandb
# uv run -m model.models --wandb