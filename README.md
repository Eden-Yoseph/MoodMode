# Mood Mode — Emotion-Aware Conversational AI

**Mood Mode** is a responsive AI chat application that dynamically adapts its conversational style based on the user’s current emotional state and preferred interaction mode. Leveraging Google Gemini’s advanced language models, Mood Mode delivers nuanced, contextually aware responses that elevate the user experience beyond traditional chatbots.

---

## Features

- **Dynamic Emotional Context:** Select from multiple moods (e.g., joyful, anxious, overwhelmed) to influence response tone and content.
- **Configurable Interaction Modes:** Choose styles ranging from gentle support and constructive feedback to analytical and silent presence.
- **Adaptive Prompt Suggestions:** Contextual conversation starters trigger during idle moments to encourage engagement.
- **User Feedback Loop:** Collects in-session feedback to refine response relevance and quality.
- **Safety & Moderation:** Real-time detection of sensitive or risky inputs ensures safe and responsible interactions.
- **Persistent Local Sessions:** Maintains chat history and feedback using local storage for continuity without backend complexity.

---

## Technology Stack

| Layer            | Technology                             |
|------------------|--------------------------------------|
| Frontend         | React (Functional Components + Hooks)|
| Backend          | FastAPI (Python)                      |
| AI Integration   | Google Gemini API                     |
| State Management | React useState, useEffect             |
| HTTP Client      | Axios                                |
| Environment      | `.env` configuration for API keys    |

---

## Architecture Overview

Mood Mode’s architecture cleanly separates concerns:

- **Frontend:** React SPA managing user input, UI state, and real-time chat updates with smooth animations.
- **Backend:** FastAPI server mediates between the frontend and Google Gemini API, handling API requests, content moderation, session management, and feedback processing.
- **AI Service:** Encapsulates calls to Google Gemini, injecting mood and mode parameters to customize AI responses.

This modular design ensures maintainability, scalability, and easy extensibility.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Access to Google Gemini API and valid API credentials

### Setup

1. Clone the repository and navigate to the backend folder:

    ```bash
    cd backend
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure environment variables:

    - Copy `.env.example` to `.env`
    - Update `.env` with your Gemini API key and relevant settings

5. Start the backend server:

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

6. Serve or open the frontend:

    - Open `frontend/index.html` directly in a modern browser, or
    - Serve with a simple HTTP server (e.g., `python -m http.server 3000`)

---

## Design Considerations & Challenges

- **Contextual Understanding:** Integrating mood and mode parameters into AI prompts to generate meaningful, empathetic responses without losing conversational flow.
- **User Safety:** Implemented keyword detection and content moderation to address sensitive topics gracefully.
- **Real-Time Feedback:** Enabled in-chat feedback collection, facilitating iterative improvements without interrupting user experience.
- **Lightweight Persistence:** Leveraged local storage to maintain session continuity without a complex backend database.
- **API Integration:** Abstracted Google Gemini calls to accommodate future API provider changes with minimal code updates.

---

## Roadmap & Future Enhancements

- Expand multilingual support to increase accessibility.
- Introduce persistent backend storage for robust session history.
- Add voice recognition and speech synthesis for richer interaction modes.
- Implement advanced sentiment analysis for automated mood detection.
- Dockerize for containerized deployment and cloud scalability.

---

## Security & Privacy

- API keys are securely stored in environment variables, never exposed in client code.
- User conversation data remains local by default, protecting privacy.
- Content moderation mechanisms actively mitigate harmful or unsafe conversations.

---

## Contribution & Support

Contributions are welcome through issues and pull requests. For questions or support, please open an issue or contact the maintainer.

---

## License

Distributed under the MIT License.

---

## Contact

Eden Yoseph  
[LinkedIn](https://linkedin.com/in/edenyoseph)
