# Start Backend (Terminal 1)
cd c:\AIIMGS
uvicorn api.main:app --reload --port 8000

# Start Frontend (Terminal 2)
cd c:\AIIMGS\frontend
npm run dev

# Access:
# - Frontend: http://localhost:3000
# - API Docs: http://localhost:8000/docs
# - Backend: http://localhost:8000

# Default login:
# Username: admin
# Password: admin123
