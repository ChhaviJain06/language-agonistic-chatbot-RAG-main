#!/usr/bin/env bash
# One-time setup of a fresh Oracle Cloud Ubuntu VM: Docker + the OS firewall.
# Oracle's Ubuntu images ship with iptables rules that block everything except SSH, so
# opening ports only in the Oracle console is NOT enough — this script opens them here too.
# Run on the VM:  bash oracle_setup.sh
set -euo pipefail

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent rsync

open_tcp() { sudo iptables -C INPUT -p tcp --dport "$1" -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 -p tcp --dport "$1" -j ACCEPT; }
open_udp() { sudo iptables -C INPUT -p udp --dport "$1" -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 -p udp --dport "$1" -j ACCEPT; }

open_tcp 80       # HTTP (certificate check + redirect)
open_tcp 443      # HTTPS (site + LiveKit signalling)
open_tcp 7881     # LiveKit WebRTC over TCP (fallback)
open_udp 50000:50100   # LiveKit WebRTC media

sudo netfilter-persistent save
echo "Done. Log out and back in once so the docker group applies."
