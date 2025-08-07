# main.py - FastAPI Backend for Emotion-Aware AI Chat

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import openai
import os
import logging
import time
import json
from datetime import datetime
import re
from textblob import TextBlob  # For sentiment analysis backup
import asyncio
from enum import Enum
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Emotion-Aware AI Chat API",
    description="AI chat service that adapts responses based on user's emotional state and preferred interaction mode",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import google.generativeai as genai

# Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Authentication (optional)
security = HTTPBearer(auto_error=False)

# Enums for validation
class MoodType(str, Enum):
    neutral = "neutral"
    joyful = "joyful"
    sad = "sad"
    anxious = "anxious"
    angry = "angry"
    numb = "numb"
    confused = "confused"
    overwhelmed = "overwhelmed"
    hopeful = "hopeful"
    frustrated = "frustrated"

class ModeType(str, Enum):
    gentle = "gentle"
    honest = "honest"
    constructive = "constructive"
    silent = "silent"
    listen = "listen"
    analytical = "analytical"
    therapist = "therapist"

# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    mood: MoodType = Field(..., description="User's current emotional state")
    mode: ModeType = Field(..., description="Preferred AI response style")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous messages for context")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Message too short')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    mood: str
    mode: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    safety_flagged: bool = False
    confidence_score: Optional[float] = None

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    feedback_type: str = Field(..., regex="^(helpful|confused|unhelpful)$")
    additional_notes: Optional[str] = None

# Detailed System Prompts for Each Mode
MODE_PROMPTS = {
    "gentle": """You are a warm, compassionate companion providing emotional support. Your role is to:
- Always validate the user's feelings as legitimate and understandable
- Use gentle, nurturing language that feels like a caring friend
- Focus on emotional safety and making the user feel heard and less alone
- Avoid giving direct advice unless explicitly asked
- Prioritize comfort and emotional validation over problem-solving
- Use phrases like "That sounds really difficult" or "Your feelings make complete sense"
- Be patient and never rush the user toward solutions
Remember: Your primary goal is to make the user feel emotionally supported and less alone.""",

    "honest": """You are a direct but empathetic advisor who prioritizes truth and clarity. Your role is to:
- Provide honest, constructive feedback while maintaining deep empathy
- Point out patterns, inconsistencies, or blind spots you notice - but gently
- Help users see situations more clearly without sugarcoating
- Balance truth-telling with emotional sensitivity
- Ask clarifying questions to help users think more deeply
- Use phrases like "I notice..." or "From what you're sharing, it seems..."
- Be straightforward about difficult truths, but deliver them with care
Remember: Your goal is helpful honesty that serves the user's long-term growth.""",

    "constructive": """You are a solution-focused mentor helping users move forward. Your role is to:
- Help identify concrete, actionable steps the user can take
- Break down overwhelming problems into manageable pieces
- Ask questions that guide users toward their own solutions
- Focus on what the user can control and influence
- Suggest practical strategies and coping mechanisms
- Balance empathy with forward momentum
- Use phrases like "What would help most right now?" or "What's one small step you could take?"
- Help users shift from feeling stuck to feeling empowered
Remember: Your goal is practical progress while honoring their emotional experience.""",

    "silent": """You are a quiet, mindful presence offering comfort through minimal but meaningful responses. Your role is to:
- Provide brief, gentle acknowledgments like "I'm here with you" or "I understand"
- Use very few words but make each one count
- Focus on presence over advice or analysis
- Sometimes respond with just "..." to indicate quiet companionship
- Only elaborate when directly asked specific questions
- Honor the power of silence and simple witness
- Use phrases like "Sitting with you in this" or "I hear you"
Remember: Sometimes the most healing response is simply being present without trying to fix or change anything.""",

    "listen": """You are focused purely on listening and reflecting back what you hear. Your role is to:
- Reflect back the user's words without interpretation or advice
- Use phrases like "I hear you saying..." or "It sounds like..."
- Ask gentle follow-up questions like "Tell me more about that"
- Avoid giving solutions, analysis, or guidance unless specifically requested
- Help the user feel truly heard and understood
- Mirror their language and emotional tone
- Focus on acknowledgment over action
Remember: Your job is to be a perfect listener who helps users feel heard and understood.""",

    "analytical": """You are a logical, structured thinker helping users analyze situations objectively. Your role is to:
- Break down complex emotions or situations into understandable components
- Help identify patterns, causes, and effects
- Ask probing questions that lead to insights
- Use logical frameworks while remaining emotionally aware
- Help users think through problems systematically
- Identify cognitive distortions or unhelpful thought patterns
- Balance rational analysis with emotional intelligence
Remember: Your goal is clarity and understanding through structured thinking.""",

    "therapist": """You are a professional counselor using evidence-based therapeutic approaches. Your role is to:
- Use active listening and reflective techniques
- Ask open-ended questions that promote self-discovery
- Help users explore their thoughts, feelings, and behaviors
- Identify patterns and connections between past and present
- Suggest gentle exercises for emotional regulation when appropriate
- Maintain professional boundaries while being deeply empathetic
- Use therapeutic language and techniques (but always remind users you're AI, not a replacement for human therapy)
Remember: Your goal is to facilitate insight and emotional growth through professional therapeutic approaches."""
}

# Safety Configuration
RISK_KEYWORDS = [
    'harm myself', 'hurt myself', 'kill myself', 'suicide', 'suicidal',
    'worthless', 'hopeless', 'can\'t go on', 'end it all', 'better off dead',
    'no point', 'hate myself', 'want to die', 'give up completely',
    'self harm', 'cut myself', 'overdose', 'jump off'
]

SAFETY_RESPONSE = """I'm really concerned about what you're sharing. These feelings sound incredibly painful and overwhelming. While I'm here to listen and support you, I think it would be really helpful for you to talk with someone who is specifically trained for these moments - like a counselor, therapist, or crisis helpline.

In the US, you can:
- Call 988 (Suicide & Crisis Lifeline) - available 24/7
- Text "HELLO" to 741741 (Crisis Text Line)
- Go to your nearest emergency room if you're in immediate danger

Your feelings are valid, your pain is real, and you deserve support. Would you like to talk about finding some professional help, or is there anything else I can do to support you right now?"""

# Utility Functions
class SafetyService:
    @staticmethod
    def detect_risk(text: str) -> bool:
        """Detect potentially harmful content in user messages."""
        normalized_text = text.lower()
        return any(keyword in normalized_text for keyword in RISK_KEYWORDS)
    
    @staticmethod
    def validate_input_length(text: str) -> bool:
        """Validate input length constraints."""
        return 1 <= len(text.strip()) <= 2000
    
    @staticmethod
    def moderate_content(text: str) -> Dict[str, Any]:
        """Basic content moderation."""
        return {
            'is_safe': not SafetyService.detect_risk(text),
            'risk_level': 'high' if SafetyService.detect_risk(text) else 'low',
            'requires_intervention': SafetyService.detect_risk(text)
        }

class AnalyticsService:
    @staticmethod
    def log_interaction(user_message: str, ai_response: str, mood: str, mode: str, 
                       user_id: Optional[str] = None, safety_flagged: bool = False):
        """Log interaction for analytics (implement your preferred storage)."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id or 'anonymous',
            'mood': mood,
            'mode': mode,
            'input_length': len(user_message),
            'output_length': len(ai_response),
            'safety_flagged': safety_flagged,
            'session_id': f"{user_id or 'anon'}_{int(time.time())}"
        }
        
        # For demo purposes, just log to console
        # In production, save to database or analytics service
        logger.info(f"Interaction logged: {json.dumps(log_entry)}")
        return log_entry

class MoodDetectionService:
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        """Backup sentiment analysis using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    @staticmethod
    def suggest_mood(text: str) -> Optional[str]:
        """Suggest mood based on text analysis if user didn't specify."""
        sentiment = MoodDetectionService.analyze_sentiment(text)
        
        # Simple mood suggestion based on polarity
        if sentiment['polarity'] > 0.3:
            return 'joyful'
        elif sentiment['polarity'] < -0.3:
            return 'sad'
        elif sentiment['subjectivity'] > 0.7:
            return 'confused'
        else:
            return 'neutral'

# AI Service
class AIService:
    @staticmethod
    async def generate_response(user_message: str, mood: str, mode: str, 
                               conversation_history: List[ChatMessage] = None) -> Dict[str, Any]:
        """Generate AI response using OpenAI with mode-specific prompts."""
        
        # Get mode-specific system prompt
        system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS['gentle'])
        
        # Add mood context to system prompt
        mood_context = f"\n\nIMPORTANT: The user is currently feeling {mood}. Adjust your response tone and approach accordingly while maintaining your {mode} mode characteristics."
        full_system_prompt = system_prompt + mood_context
        
        # Prepare conversation history
        messages = [{"role": "system", "content": full_system_prompt}]
        
        # Add recent conversation history for context (last 6 messages)
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        try:
            # Call OpenAI API
            response = await asyncio.create_task(
                AIService._call_openai(messages)
            )
            
            return {
                'response': response['choices'][0]['message']['content'].strip(),
                'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                'model_used': response.get('model', 'gpt-3.5-turbo'),
                'finish_reason': response['choices'][0].get('finish_reason', 'stop')
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Fallback to mock response
            return AIService._generate_fallback_response(user_message, mood, mode)
    
    @staticmethod
    async def _call_openai(messages: List[Dict]) -> Dict:
        """Make actual OpenAI API call."""
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            frequency_penalty=0.3,
            presence_penalty=0.3
        )
    
    @staticmethod
    def _generate_fallback_response(user_message: str, mood: str, mode: str) -> Dict[str, Any]:
        """Generate fallback response when OpenAI is unavailable."""
        
        fallback_responses = {
            'gentle': [
                f"I can hear that you're feeling {mood}, and I want you to know that what you're experiencing is completely valid. I'm here with you.",
                f"Thank you for sharing these {mood} feelings with me. Your emotions deserve to be acknowledged and respected.",
                f"It takes courage to express when you're feeling {mood}. I'm here to listen and support you through this."
            ],
            'honest': [
                f"I notice you're experiencing {mood} feelings about this situation. What do you think is at the core of what you're going through?",
                f"Your {mood} response tells us something important. What patterns do you see in this situation?",
                f"Being direct with you - feeling {mood} about this makes complete sense given what you've shared."
            ],
            'constructive': [
                f"Given that you're feeling {mood}, what's one small step that might help you move forward today?",
                f"I can see why this situation would leave you feeling {mood}. What aspects feel most within your control?",
                f"This {mood} feeling is information about what matters to you. How might you use this insight constructively?"
            ],
            'silent': [
                "I'm here with you.",
                "Sitting quietly with you in this moment.",
                "I hear you.",
                "..."
            ],
            'listen': [
                "I'm listening.",
                f"I hear you saying you're feeling {mood} about this.",
                "Tell me more.",
                "What else is there?"
            ],
            'analytical': [
                f"Looking at this analytically, your {mood} response seems to have specific triggers. What might those be?",
                f"Let's break down the components: what facts are contributing to this {mood} feeling?",
                f"From a logical perspective, what variables are most significant in this {mood} experience?"
            ],
            'therapist': [
                f"I notice you're experiencing {mood} feelings. Can you help me understand when this started?",
                f"Your {mood} response is telling us something important about your needs. What might this feeling be communicating?",
                f"Let's explore this {mood} experience together. What thoughts typically accompany this feeling?"
            ]
        }
        
        mode_responses = fallback_responses.get(mode, fallback_responses['gentle'])
        import random
        selected_response = random.choice(mode_responses)
        
        return {
            'response': selected_response,
            'tokens_used': 0,
            'model_used': 'fallback',
            'finish_reason': 'fallback'
        }

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Emotion-Aware AI Chat API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow()
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "ai_service": "operational",
            "safety_service": "operational",
            "analytics_service": "operational"
        },
        "supported_moods": list(MoodType),
        "supported_modes": list(ModeType)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, credentials = Depends(security)):
    """Main chat endpoint that processes user messages and returns AI responses."""
    
    try:
        # Input validation
        if not SafetyService.validate_input_length(request.message):
            raise HTTPException(status_code=400, detail="Invalid message length")
        
        # Safety check
        safety_check = SafetyService.moderate_content(request.message)
        
        if safety_check['requires_intervention']:
            # Return safety response for high-risk content
            response = ChatResponse(
                response=SAFETY_RESPONSE,
                mood=request.mood.value,
                mode=request.mode.value,
                timestamp=datetime.utcnow(),
                safety_flagged=True,
                tokens_used=0
            )
            
            # Log the safety intervention
            AnalyticsService.log_interaction(
                user_message=request.message,
                ai_response=SAFETY_RESPONSE,
                mood=request.mood.value,
                mode=request.mode.value,
                user_id=request.user_id,
                safety_flagged=True
            )
            
            return response
        
        # Generate AI response
        ai_result = await AIService.generate_response(
            user_message=request.message,
            mood=request.mood.value,
            mode=request.mode.value,
            conversation_history=request.conversation_history
        )
        
        # Calculate confidence score (mock implementation)
        confidence_score = min(0.95, 0.7 + (ai_result['tokens_used'] / 1000))
        
        # Create response
        response = ChatResponse(
            response=ai_result['response'],
            mood=request.mood.value,
            mode=request.mode.value,
            timestamp=datetime.utcnow(),
            tokens_used=ai_result['tokens_used'],
            safety_flagged=False,
            confidence_score=confidence_score
        )
        
        # Log successful interaction
        AnalyticsService.log_interaction(
            user_message=request.message,
            ai_response=ai_result['response'],
            mood=request.mood.value,
            mode=request.mode.value,
            user_id=request.user_id,
            safety_flagged=False
        )
        
        return response
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Endpoint to collect user feedback on AI responses."""
    
    try:
        # Log feedback for analytics
        feedback_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': request.session_id,
            'message_id': request.message_id,
            'feedback_type': request.feedback_type,
            'additional_notes': request.additional_notes
        }
        
        # In production, save to database
        logger.info(f"Feedback received: {json.dumps(feedback_entry)}")
        
        return {
            "message": "Feedback received successfully",
            "feedback_id": f"fb_{int(time.time())}",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")

@app.get("/moods")
async def get_moods():
    """Get available mood options."""
    return {
        "moods": [
            {"value": mood.value, "label": mood.value.title()}
            for mood in MoodType
        ]
    }

@app.get("/modes")
async def get_modes():
    """Get available response modes with descriptions."""
    mode_descriptions = {
        "gentle": "Warm, validating, and comforting responses",
        "honest": "Direct, truthful feedback with empathy",
        "constructive": "Problem-solving focus with actionable advice", 
        "silent": "Minimal words, quiet companionship",
        "listen": "No advice, just acknowledgment",
        "analytical": "Logic-based, structured thinking",
        "therapist": "Professional counseling approach"
    }
    
    return {
        "modes": [
            {
                "value": mode.value,
                "label": mode.value.title(),
                "description": mode_descriptions.get(mode.value, "")
            }
            for mode in ModeType
        ]
    }

@app.get("/analytics/stats")
async def get_analytics_stats():
    """Get basic usage analytics (mock implementation)."""
    # In production, query your database for real stats
    return {
        "total_conversations": 1247,
        "popular_moods": {
            "anxious": 32,
            "sad": 28,
            "confused": 18,
            "neutral": 15,
            "overwhelmed": 7
        },
        "popular_modes": {
            "gentle": 45,
            "honest": 22,
            "constructive": 18,
            "therapist": 10,
            "analytical": 5
        },
        "feedback_distribution": {
            "helpful": 78,
            "confused": 15,
            "unhelpful": 7
        },
        "safety_interventions": 12,
        "average_conversation_length": 4.3
    }

@app.get("/prompts/suggest/{mood}/{mode}")
async def get_prompt_suggestion(mood: MoodType, mode: ModeType):
    """Get conversation starter suggestions based on mood and mode."""
    
    suggestions = {
        f"{mood.value}-{mode.value}": [
            f"What's been on your mind lately while feeling {mood.value}?",
            f"How has this {mood.value} feeling been showing up for you?",
            f"What would be most helpful to explore about feeling {mood.value}?"
        ]
    }
    
    # Add specific combinations
    specific_suggestions = {
        "sad-gentle": [
            "What's one small thing that brought you even a tiny bit of comfort today?",
            "What does this sadness feel like right now? There's no rush to feel differently.",
        ],
        "anxious-constructive": [
            "What specific worry is taking up the most mental space right now?",
            "What's one small step that might help you feel more grounded today?",
        ],
        "angry-honest": [
            "What's really at the heart of this anger? What do you need most right now?",
            "What would you say to someone who made you feel truly heard about this anger?",
        ]
    }
    
    key = f"{mood.value}-{mode.value}"
    prompts = specific_suggestions.get(key, suggestions.get(key, [
        f"What would you like to explore while feeling {mood.value}?",
        f"How can I best support you in {mode.value} mode today?",
        f"What's most important for you to talk through right now?"
    ]))
    
    return {
        "mood": mood.value,
        "mode": mode.value,
        "suggestions": prompts
    }

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "An unexpected error occurred",
        "status_code": 500,
        "timestamp": datetime.utcnow()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Emotion-Aware AI Chat API starting up...")
    logger.info(f"Supported moods: {[m.value for m in MoodType]}")
    logger.info(f"Supported modes: {[m.value for m in ModeType]}")
    logger.info("API ready to accept requests")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )