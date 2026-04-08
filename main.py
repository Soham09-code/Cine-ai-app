import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "962214dfd0adcb682d902d14d6797fdc") # Default from HTML
BASE_TMDB_URL = "https://api.themoviedb.org/3"

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

app = FastAPI(title="CineAI Backend")

# CORS Middleware for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoodRequest(BaseModel):
    mood: str

class MovieAnalysisRequest(BaseModel):
    title: str
    overview: str
    genres: List[str]

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "cineai_v2.html"
    if not html_path.exists():
        return "CineAI Logic is running, but frontend (cineai_v2.html) was not found in the root."
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/status")
async def status():
    return {"status": "CineAI API is running", "ai_enabled": model is not None}

@app.post("/api/mood")
async def process_mood(request: MoodRequest):
    """
    Translates natural language mood into TMDB discovery parameters using Gemini.
    """
    if not model:
        # Fallback if no Gemini key
        return {"with_genres": "", "description": "Mood analysis requires API key."}

    prompt = f"""
    Translate the following movie mood/query into TMDB technical filters.
    User Query: "{request.mood}"

    Respond ONLY in JSON format with:
    - "with_genres": A comma-separated string of TMDB genre IDs.
    - "without_genres": Excluded genre IDs.
    - "sort_by": One of [popularity.desc, vote_average.desc, primary_release_date.desc].
    - "ai_explanation": A 1-sentence catchy explanation of why these filters match.

    TMDB Genre IDs Reference:
    Action: 28, Adventure: 12, Animation: 16, Comedy: 35, Crime: 80, Documentary: 99, 
    Drama: 18, Family: 10751, Fantasy: 14, History: 36, Horror: 27, Music: 10402, 
    Mystery: 9648, Romance: 10749, Science Fiction: 878, TV Movie: 10770, 
    Thriller: 53, War: 10752, Western: 37.
    """
    
    try:
        response = model.generate_content(prompt)
        # Handle potential markdown formatting in response
        text = response.text.strip().replace('```json', '').replace('```', '')
        import json
        return json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_movie(request: MovieAnalysisRequest):
    """
    Generates a personalized "AI Commentary" for a specific movie.
    """
    if not model:
        return {"analysis": "Connect Gemini API to see AI insights."}

    prompt = f"""
    Analyze this movie and provide a "Director's Note" style narrative (2-3 sentences max).
    Title: {request.title}
    Genres: {", ".join(request.genres)}
    Overview: {request.overview}

    Focus on the emotional tone and why a movie-lover would appreciate its cinematic choices.
    """

    try:
        response = model.generate_content(prompt)
        return {"analysis": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
