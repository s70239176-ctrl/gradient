#!/usr/bin/env bash
# setup.sh — One-command local dev setup for NFT Floor Forecaster
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ◈ NFT Floor Forecaster — Setup"
echo "  ─────────────────────────────────────────────────────"
echo -e "${NC}"

# Check deps
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js 18+ required"; exit 1; }

# Python deps
echo -e "${YELLOW}[1/4] Installing Python dependencies...${NC}"
pip install -r backend/requirements.txt -q

# Frontend deps
echo -e "${YELLOW}[2/4] Installing frontend dependencies...${NC}"
cd frontend && npm install --legacy-peer-deps --silent && cd ..

# Run tests
echo -e "${YELLOW}[3/4] Running test suite...${NC}"
python -m pytest tests/ -q --tb=short
echo -e "${GREEN}  ✓ All tests passed${NC}"

# Copy env
if [ ! -f .env ]; then
  cp .env.example .env
  echo -e "${YELLOW}[4/4] Created .env from .env.example${NC}"
  echo -e "      Edit .env to add your RESERVOIR_API_KEY for live data"
else
  echo -e "${YELLOW}[4/4] .env already exists — skipping${NC}"
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "  Start the app:"
echo -e "  ${CYAN}# Terminal 1 — Backend${NC}"
echo "  cd backend && uvicorn main:app --reload --port 8000"
echo ""
echo -e "  ${CYAN}# Terminal 2 — Frontend (dev server)${NC}"
echo "  cd frontend && npm run dev"
echo ""
echo "  Then open: http://localhost:5173"
echo ""
echo "  Or with Docker:"
echo "  docker compose up --build"
echo ""
echo "  Train a model (optional):"
echo "  python train.py --collections 8 --epochs 20"
echo ""
