#!/usr/bin/env bash
# ============================================================================ #
#                    DonaNet — Environment Setup Script
# ============================================================================ #
# Sets up a Python virtual environment with uv and installs the dependencies
# needed to run configure.py.
#
# Usage:
#   ./setup.sh
# ============================================================================ #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                  DonaNet — Environment Setup                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

check_uv() {
    if command -v uv &> /dev/null; then
        print_success "uv is already installed: $(uv --version)"
        return 0
    else
        return 1
    fi
}

install_uv() {
    print_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    for p in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [[ -f "$p/uv" ]] && export PATH="$p:$PATH" && break
    done
    if command -v uv &> /dev/null; then
        print_success "uv installed: $(uv --version)"
    else
        print_error "Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
}

create_venv() {
    print_info "Creating virtual environment in ${VENV_DIR}..."
    if [[ -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment already exists — recreating..."
        rm -rf "$VENV_DIR"
    fi
    uv venv "$VENV_DIR"
    print_success "Virtual environment created"
}

install_dependencies() {
    print_info "Installing dependencies (including configure group)..."
    source "$VENV_DIR/bin/activate"
    uv sync --group configure
    print_success "Dependencies installed"
}

print_next_steps() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                       Setup complete!                            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    print_info "Run the configuration wizard:"
    echo ""
    echo "  source .venv/bin/activate"
    echo "  python configure.py interactive"
    echo ""
    print_info "Available commands:"
    echo "  configure.py interactive           Full interactive wizard"
    echo "  configure.py quick --profile prod  Quick setup with defaults"
    echo "  configure.py quick --profile prod --gpu  With GPU support"
    echo "  configure.py list-profiles         List existing profiles"
    echo "  configure.py delete-profile NAME   Delete a profile"
    echo "  configure.py check                 Check system requirements"
    echo ""
}

print_header
cd "$SCRIPT_DIR"

check_uv || install_uv
create_venv
install_dependencies
print_next_steps
