#!/usr/bin/env bash
# =============================================================================
# LLM Serving Engine Installer
# Supports: vLLM | SGLang | Ollama
# Usage: chmod +x install.sh && ./install.sh
# =============================================================================

set -e  # Exit on any error

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Banner ─────────────────────────────────────────────────────────────────────
print_banner() {
    echo ""
    echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║       🚀 Concurrent LLM Serving Installer            ║${NC}"
    echo -e "${BOLD}${BLUE}║         vLLM  •  SGLang  •  Ollama                   ║${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ── Helpers ────────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; exit 1; }
step()    { echo -e "\n${BOLD}${PURPLE}▶ $1${NC}"; }

# ── Check for uv ──────────────────────────────────────────────────────────────
install_uv() {
    if ! command -v uv &> /dev/null; then
        step "Installing uv (fast Python package manager)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to current shell session
        export PATH="$HOME/.local/bin:$PATH"
        success "uv installed"
    else
        success "uv already installed ($(uv --version))"
    fi
}

# ── Check Python 3.12 ─────────────────────────────────────────────────────────
check_python() {
    if command -v python3.12 &> /dev/null; then
        success "Python 3.12 found"
    else
        warn "Python 3.12 not found. Trying python3..."
        if command -v python3 &> /dev/null; then
            PY_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
            warn "Using python3 ($PY_VER). Recommended: 3.12"
            PYTHON_BIN="python3"
        else
            error "No Python 3 found. Please install Python 3.12 first."
        fi
    fi
    PYTHON_BIN="${PYTHON_BIN:-python3.12}"
}

# ── Check NVIDIA GPU ───────────────────────────────────────────────────────────
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        success "GPU detected: ${GPU_NAME}"
    else
        warn "No NVIDIA GPU detected (nvidia-smi not found)."
        warn "vLLM and SGLang require a CUDA-capable GPU."
        warn "Ollama can run on CPU but will be slow."
    fi
}

# ── Print speed comparison ─────────────────────────────────────────────────────
print_comparison() {
    echo ""
    echo -e "${BOLD}📊 Benchmark Results (Qwen3.5-0.8B, 16 concurrent requests):${NC}"
    echo -e "  ${PURPLE}SGLang${NC}  ██  ${GREEN}2.47s${NC}   — fastest, RadixAttention prefix cache"
    echo -e "  ${BLUE}vLLM${NC}    ██████████████████████████████  ${YELLOW}11.26s${NC}  — stable, PagedAttention"
    echo -e "  ${YELLOW}Ollama${NC}  (4 requests only) ${RED}134.72s${NC} — sequential, for local dev"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Installer: vLLM
# ─────────────────────────────────────────────────────────────────────────────
install_vllm() {
    VENV_DIR="$HOME/vllm-env"
    step "Setting up vLLM environment at ${VENV_DIR}"

    install_uv
    check_python

    step "Creating Python virtual environment"
    $PYTHON_BIN -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    success "Virtual environment activated: ${VENV_DIR}"

    step "Upgrading pip, setuptools, wheel"
    pip install --upgrade pip setuptools wheel --quiet

    step "Installing vLLM (nightly via uv)"
    info "This may take 5–15 minutes depending on your connection..."
    uv pip install vllm \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly

    success "vLLM installed successfully!"

    echo ""
    echo -e "${BOLD}${GREEN}✅ vLLM Installation Complete!${NC}"
    echo ""
    echo -e "${BOLD}To start the server:${NC}"
    echo -e "${CYAN}source ${VENV_DIR}/bin/activate${NC}"
    echo -e "${CYAN}vllm serve Qwen/Qwen3.5-0.8B \\${NC}"
    echo -e "${CYAN}    --port 8000 \\${NC}"
    echo -e "${CYAN}    --tensor-parallel-size 1 \\${NC}"
    echo -e "${CYAN}    --max-model-len 32768 \\${NC}"
    echo -e "${CYAN}    --gpu-memory-utilization 0.75 \\${NC}"
    echo -e "${CYAN}    --reasoning-parser qwen3${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Remember: Send a warm-up request before real traffic!${NC}"
    echo -e "${YELLOW}   First cold request triggers CUDA graph compilation.${NC}"
    echo ""
    echo -e "${BOLD}Then run the test:${NC}"
    echo -e "${CYAN}pip install langchain-openai${NC}"
    echo -e "${CYAN}python vllm_concurrent_test.py${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Installer: SGLang
# ─────────────────────────────────────────────────────────────────────────────
install_sglang() {
    VENV_DIR="$HOME/sglang-env"
    step "Setting up SGLang environment at ${VENV_DIR}"

    install_uv
    check_python

    step "Creating Python virtual environment"
    $PYTHON_BIN -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    success "Virtual environment activated: ${VENV_DIR}"

    step "Upgrading pip"
    pip install --upgrade pip --quiet

    step "Installing SGLang from GitHub (latest)"
    info "This may take 5–15 minutes depending on your connection..."
    uv pip install \
        'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'

    success "SGLang installed successfully!"

    echo ""
    echo -e "${BOLD}${GREEN}✅ SGLang Installation Complete!${NC}"
    echo ""
    echo -e "${BOLD}To start the server:${NC}"
    echo -e "${CYAN}source ${VENV_DIR}/bin/activate${NC}"
    echo -e "${CYAN}python -m sglang.launch_server \\${NC}"
    echo -e "${CYAN}    --model-path Qwen/Qwen3.5-0.8B \\${NC}"
    echo -e "${CYAN}    --port 8000 \\${NC}"
    echo -e "${CYAN}    --tp-size 1 \\${NC}"
    echo -e "${CYAN}    --mem-fraction-static 0.8 \\${NC}"
    echo -e "${CYAN}    --context-length 32768 \\${NC}"
    echo -e "${CYAN}    --attention-backend triton${NC}"
    echo ""
    echo -e "${GREEN}✨ No warm-up needed — SGLang is ready immediately on startup!${NC}"
    echo ""
    echo -e "${BOLD}Then run the test:${NC}"
    echo -e "${CYAN}pip install langchain-openai${NC}"
    echo -e "${CYAN}python sglang_concurrent_test.py${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Installer: Ollama
# ─────────────────────────────────────────────────────────────────────────────
install_ollama() {
    step "Installing Ollama"

    if command -v ollama &> /dev/null; then
        success "Ollama already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
    else
        info "Downloading and installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        success "Ollama installed"
    fi

    step "Pulling Qwen3.5-0.8B model"
    info "Starting Ollama server in background..."

    # Start ollama serve if not running
    if ! pgrep -x "ollama" > /dev/null; then
        ollama serve &
        OLLAMA_PID=$!
        info "Ollama server started (PID: ${OLLAMA_PID})"
        sleep 3  # Wait for server to be ready
    else
        info "Ollama server already running"
    fi

    info "Pulling Qwen3.5:0.8b model (this downloads the GGUF file)..."
    ollama pull qwen3.5:0.8b

    success "Model pulled successfully!"

    echo ""
    echo -e "${BOLD}${GREEN}✅ Ollama Installation Complete!${NC}"
    echo ""
    echo -e "${BOLD}To start the server manually:${NC}"
    echo -e "${CYAN}ollama serve${NC}"
    echo ""
    echo -e "${BOLD}To run the model interactively:${NC}"
    echo -e "${CYAN}ollama run qwen3.5:0.8b${NC}"
    echo ""
    echo -e "${BOLD}Then run the concurrent test:${NC}"
    echo -e "${CYAN}pip install langchain-ollama${NC}"
    echo -e "${CYAN}python ollama_concurrent_test.py${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Note: Ollama processes requests sequentially by default.${NC}"
    echo -e "${YELLOW}   For limited concurrency: export OLLAMA_NUM_PARALLEL=4${NC}"
    echo -e "${YELLOW}   For production, consider vLLM or SGLang instead.${NC}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Install test dependencies
# ─────────────────────────────────────────────────────────────────────────────
install_test_deps() {
    step "Installing test script dependencies"
    pip install langchain-openai langchain-ollama --quiet
    success "Test dependencies installed (langchain-openai, langchain-ollama)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main Menu
# ─────────────────────────────────────────────────────────────────────────────
main() {
    print_banner
    check_gpu
    print_comparison

    echo -e "${BOLD}Which engine would you like to install?${NC}"
    echo ""
    echo -e "  ${PURPLE}[1]${NC} SGLang  — ${GREEN}Fastest${NC} (2.47s for 16 requests) | RadixAttention | Production"
    echo -e "  ${BLUE}[2]${NC} vLLM    — ${YELLOW}Stable${NC} (11.26s for 16 requests) | PagedAttention | Industry Standard"
    echo -e "  ${YELLOW}[3]${NC} Ollama  — ${CYAN}Easiest${NC} (local dev, single user) | GGUF quantization"
    echo -e "  ${RED}[4]${NC} Exit"
    echo ""
    read -rp "$(echo -e ${BOLD})Enter choice [1-4]: $(echo -e ${NC})" choice

    case "$choice" in
        1)
            echo ""
            echo -e "${PURPLE}→ Installing SGLang...${NC}"
            install_sglang
            install_test_deps
            ;;
        2)
            echo ""
            echo -e "${BLUE}→ Installing vLLM...${NC}"
            install_vllm
            install_test_deps
            ;;
        3)
            echo ""
            echo -e "${YELLOW}→ Installing Ollama...${NC}"
            install_ollama
            pip install langchain-ollama --quiet 2>/dev/null || true
            ;;
        4)
            echo -e "${CYAN}Bye! 👋${NC}"
            exit 0
            ;;
        *)
            error "Invalid choice. Please run the script again and choose 1–4."
            ;;
    esac

    echo ""
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║   🎉 Installation complete! Check README.md for      ║${NC}"
    echo -e "${BOLD}${GREEN}║      algorithm explanations and benchmark results.    ║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

main "$@"
