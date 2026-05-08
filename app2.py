# app.py
import streamlit as st
import re
import textwrap
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import time
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import math
import collections

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Set page configuration
st.set_page_config(
    page_title="Summarease",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        background-color: #F8F9FA;
        border: 2px solid #E9ECEF;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .transcript-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.9em;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-left: 4px solid #FF4B4B;
    }
    .chat-message.assistant {
        background-color: #475063;
        border-left: 4px solid #1f77b4;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# AI SUMMARIZATION CLASS USING TRANSFORMERS
class AISummarizer:
    def __init__(self):
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False

    def load_model(self):
        """Load the transformer model for summarization"""
        if self.model_loaded:
            return
            
        with st.spinner(" Loading model (this may take a minute)..."):
            try:
                # Using a smaller, faster model for summarization
                model_name = "facebook/bart-large-cnn"
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Create pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                
                self.model_loaded = True
                st.success(" Model loaded successfully!")
                
            except Exception as e:
                st.error(f" Failed to load model: {e}")
                # Fallback to basic summarization
                self.model_loaded = False

    def _clean_transcript_text(self, text: str) -> str:
        """Clean transcript text for better summarization"""
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause] etc
        text = re.sub(r'\(.*?\)', '', text)  # Remove (music) etc
        text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
        text = re.sub(r'\d+:\d+', '', text)  # Remove timestamps
        return text.strip()

    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        """Split text into chunks that fit the model's context window"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed the limit, save current chunk and start new one
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def summarize_with_ai(self, text: str, summary_ratio: float = 0.3) -> str:
        """Generate summary using transformer model"""
        if not text or len(text.strip()) < 100:
            return "Not enough content to summarize."
        
        # Clean the text first
        clean_text = self._clean_transcript_text(text)
        
        # If text is short, summarize directly
        if len(clean_text) < 1000:
            try:
                summary = self.summarizer(
                    clean_text,
                    max_length=int(len(clean_text) * summary_ratio),
                    min_length=100,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                return summary
            except Exception as e:
                st.warning(f" Summarization failed, using fallback: {e}")
                return self._fallback_summarize(clean_text, summary_ratio)
        
        # For longer texts, chunk and summarize
        try:
            chunks = self._chunk_text(clean_text)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 100:  # Only summarize meaningful chunks
                    chunk_summary = self.summarizer(
                        chunk,
                        max_length=200,
                        min_length=50,
                        do_sample=False,
                        truncation=True
                    )[0]['summary_text']
                    summaries.append(chunk_summary)
            
            # Combine chunk summaries and create final summary
            combined_text = " ".join(summaries)
            if len(combined_text) > 500:
                final_summary = self.summarizer(
                    combined_text,
                    max_length=300,
                    min_length=150,
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
            else:
                final_summary = combined_text
                
            return final_summary
            
        except Exception as e:
            st.warning(f" Summarization failed, using fallback: {e}")
            return self._fallback_summarize(clean_text, summary_ratio)

    def _fallback_summarize(self, text: str, summary_ratio: float = 0.3) -> str:
        """Fallback summarization when AI model fails"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 3:
            return text[:400] + "..." if len(text) > 400 else text
        
        # Take first, middle, and last sentences
        num_sentences = max(3, min(5, int(len(sentences) * summary_ratio)))
        selected_indices = [0]  # First sentence
        
        # Add some middle sentences
        middle_start = len(sentences) // 3
        for i in range(1, num_sentences - 1):
            idx = middle_start + (i * len(sentences) // (num_sentences + 1))
            if idx < len(sentences):
                selected_indices.append(idx)
        
        # Add last sentence
        if len(sentences) > 1:
            selected_indices.append(len(sentences) - 1)
        
        selected_sentences = [sentences[i] for i in selected_indices if i < len(sentences)]
        return '. '.join(selected_sentences) + '.'

# OPTIMIZED AI Chatbot Class with Enhanced Context Understanding
class AIChatBot:
    def __init__(self):
        self.qa_pipeline = None
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.context_cache = {}

    def load_model(self):
        """Load a better Q&A model optimized for long-form content"""
        if self.model_loaded:
            return

        with st.spinner("🤖 Loading enhanced Q&A model for video content..."):
            try:
                # Using a model better suited for conversational QA and longer contexts
                model_name = "deepset/roberta-base-squad2"
                
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    max_seq_len=512,
                    doc_stride=128
                )
                
                self.model_loaded = True
                st.success("✅ Enhanced Q&A model loaded! Ready for video questions.")
                
            except Exception as e:
                st.error(f"❌ Failed to load enhanced model: {e}")
                # Fallback to simpler model
                try:
                    model_name = "distilbert-base-cased-distilled-squad"
                    self.qa_pipeline = pipeline(
                        "question-answering",
                        model=model_name,
                        tokenizer=model_name,
                        device=0 if self.device == "cuda" else -1
                    )
                    self.model_loaded = True
                    st.success("✅ Basic Q&A model loaded.")
                except Exception as e2:
                    st.error(f"❌ Complete model load failure: {e2}")
                    self.model_loaded = False

    def _find_relevant_context(self, question: str, full_transcript: str, summary: str = "") -> str:
        """Find the most relevant context from transcript for the question"""
        # Cache processed contexts to avoid recomputation
        cache_key = hash(full_transcript[:1000] + question)
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        # Combine summary and transcript for better context
        combined_text = ""
        if summary and len(summary) > 50:
            combined_text = summary + " " + full_transcript
        else:
            combined_text = full_transcript
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        
        # Preprocess question
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
        
        # Score sentences based on relevance to question
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence_lower))
            
            # Calculate relevance score
            word_overlap = len(question_words.intersection(sentence_words))
            position_score = 1.0 / (i + 1)  # Earlier sentences might be more important
            length_score = min(len(sentence.split()) / 50, 1.0)  # Optimal sentence length
            
            # Check for question patterns in sentence
            pattern_score = 0
            for q_word in question_words:
                if q_word in sentence_lower and len(q_word) > 3:
                    pattern_score += 1
            
            total_score = (word_overlap * 2) + position_score + length_score + pattern_score
            scored_sentences.append((total_score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:8]]  # Top 8 sentences
        
        # Build context
        context = " ".join(top_sentences)
        
        # Cache result
        self.context_cache[cache_key] = context[:1500]  # Limit context length
        
        return self.context_cache[cache_key]

    def _extract_key_info_from_context(self, context: str, question: str) -> List[str]:
        """Extract key information that might answer the question"""
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Keywords from question
        question_keywords = set(re.findall(r'\b\w{3,}\b', question.lower()))
        
        relevant_info = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_keywords = set(re.findall(r'\b\w{3,}\b', sentence_lower))
            
            # Check for keyword overlap
            overlap = len(question_keywords.intersection(sentence_keywords))
            if overlap >= 1 and len(sentence.strip()) > 20:
                relevant_info.append(sentence.strip())
        
        return relevant_info[:5]  # Return top 5 relevant sentences

    def answer_question(self, question: str, full_transcript: str, summary: str = "") -> str:
        """Enhanced question answering with better context understanding"""
        if not self.model_loaded:
            return "The Q&A model is still loading. Please wait a moment."

        if not full_transcript or len(full_transcript) < 100:
            return "I don't have enough transcript text to answer from."

        # Step 1: Find most relevant context
        context = self._find_relevant_context(question, full_transcript, summary)
        
        if len(context) < 50:
            context = full_transcript[:1500]  # Fallback to first part of transcript
        
        # Step 2: Use QA pipeline
        try:
            result = self.qa_pipeline({
                'context': context,
                'question': question
            })
            
            answer = result['answer'].strip()
            confidence = result['score']
            
            # Step 3: Validate and enhance answer
            if confidence < 0.05 or len(answer) < 3:
                return self._generate_comprehensive_answer(question, full_transcript, summary)
            
            # Step 4: Check if answer needs elaboration
            if len(answer.split()) < 10 and confidence < 0.3:
                # Try to get more context
                additional_info = self._extract_key_info_from_context(full_transcript, question)
                if additional_info:
                    answer += " More specifically: " + " ".join(additional_info[:2])
            
            # Step 5: Format answer
            if not answer.endswith(('.', '!', '?')):
                answer = answer.rstrip(',;:') + '.'
            
            # Add confidence indicator for low confidence answers
            if confidence < 0.2:
                answer = f"Based on the video content: {answer}"
            
            return answer
            
        except Exception as e:
            return self._generate_comprehensive_answer(question, full_transcript, summary)

    def _generate_comprehensive_answer(self, question: str, full_transcript: str, summary: str = "") -> str:
        """Generate a comprehensive answer when QA model has low confidence"""
        # Extract relevant sentences
        question_lower = question.lower()
        sentences = re.split(r'(?<=[.!?])\s+', full_transcript)
        
        # Categorize question type
        question_types = {
            'what': ['what is', 'what are', 'what does', 'what do'],
            'how': ['how to', 'how does', 'how do', 'how can'],
            'why': ['why is', 'why are', 'why does', 'why do'],
            'who': ['who is', 'who are', 'who does'],
            'when': ['when is', 'when are', 'when does', 'when do']
        }
        
        # Find relevant sentences
        relevant_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 30:
                continue
                
            sentence_lower = sentence.lower()
            
            # Check for direct matches
            for q_type in question_types:
                for pattern in question_types[q_type]:
                    if pattern in question_lower and any(word in sentence_lower for word in question_lower.split()[1:]):
                        relevant_sentences.append(sentence)
                        break
            
            # Check for keyword overlap
            question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
            sentence_words = set(re.findall(r'\b\w{3,}\b', sentence_lower))
            if len(question_words.intersection(sentence_words)) >= 2:
                relevant_sentences.append(sentence)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for s in relevant_sentences:
            if s not in seen:
                seen.add(s)
                unique_sentences.append(s)
        
        # If we found relevant sentences, construct answer
        if unique_sentences:
            if len(unique_sentences) == 1:
                return f"The video mentions: {unique_sentences[0]}"
            else:
                return f"Based on the video: {' '.join(unique_sentences[:3])}"
        
        # Fallback: Use summary or general answer
        if summary and len(summary) > 50:
            return f"The main points from the video are: {summary[:300]}..."
        
        # Check if it's a generic question
        generic_responses = {
            'summary': "Could you provide a summary?",
            'main points': "What are the main points?",
            'key takeaways': "What are the key takeaways?",
            'about': "What is this video about?",
            'explain': "Can you explain the content?"
        }
        
        for key, response in generic_responses.items():
            if key in question_lower:
                if summary:
                    return f"The video discusses: {summary[:200]}..."
                else:
                    return "The video covers various topics discussed in the transcript."
        
        return "I couldn't find specific information about that in the video. The content may not directly address your question."

    def generate_conversational_response(self, question: str, full_transcript: str, summary: str = "") -> str:
        """Generate a conversational, helpful response"""
        # First, get a direct answer
        direct_answer = self.answer_question(question, full_transcript, summary)
        
        # Make it more conversational
        question_lower = question.lower()
        
        # Add appropriate prefix based on question type
        if any(word in question_lower for word in ['what', 'how', 'why', 'who', 'when']):
            if direct_answer.startswith(('The video', 'Based on', 'The main')):
                return direct_answer
            elif 'could not find' in direct_answer.lower():
                return direct_answer
            else:
                return f"In the video, {direct_answer.lower() if direct_answer[0].isupper() else direct_answer}"
        
        return direct_answer

# YouTube API Manager (unchanged)
class YouTubeAPIManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self._video_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&?\n]+)',
            r'youtube\.com/embed/([^&?\n]+)',
            r'youtube\.com/v/([^&?\n]+)'
        ]

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from URL"""
        url = url.strip()
        for pattern in self._video_patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                if len(video_id) == 11:
                    return video_id
        if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
        raise ValueError(f"Could not extract video ID from URL: {url}")

    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get video details"""
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()

            if not response['items']:
                return None

            video_data = response['items'][0]
            snippet = video_data['snippet']
            statistics = video_data.get('statistics', {})

            return {
                'title': snippet.get('title', 'No Title'),
                'description': snippet.get('description', ''),
                'channel': snippet.get('channelTitle', 'Unknown Channel'),
                'views': statistics.get('viewCount', 'N/A'),
                'likes': statistics.get('likeCount', 'N/A'),
                'published_at': snippet.get('publishedAt', 'N/A'),
            }
        except Exception as e:
            st.error(f" YouTube API Error: {e}")
            return None

    def get_video_transcript(self, video_id: str) -> Tuple[str, str, str]:
        """Get video transcript using the new API with fetch() function"""
        try:
            st.info(" Searching for transcript...")
            
            # Initialize the API
            ytt_api = YouTubeTranscriptApi()
            
            # Method 1: Try to fetch English transcript directly
            try:
                fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
                transcript_text = ' '.join([snippet.text for snippet in fetched_transcript])
                if transcript_text and len(transcript_text) > 50:
                    st.success(" Found English transcript!")
                    return transcript_text, "English transcript", "en"
            except Exception as e:
                st.warning(f" English transcript not available: {e}")
            
            # Method 2: Try to fetch any available transcript
            try:
                fetched_transcript = ytt_api.fetch(video_id)
                transcript_text = ' '.join([snippet.text for snippet in fetched_transcript])
                if transcript_text and len(transcript_text) > 50:
                    st.success(" Found auto-detected transcript!")
                    return transcript_text, "Auto-detected transcript", fetched_transcript.language_code
            except Exception as e:
                st.warning(f" Auto-detected transcript failed: {e}")
            
            # Method 3: List available transcripts and try each one
            try:
                transcript_list = ytt_api.list(video_id)
                available_languages = [transcript.language for transcript in transcript_list]
                st.info(f" Available languages: {available_languages}")
                
                for transcript in transcript_list:
                    try:
                        fetched_transcript = transcript.fetch()
                        transcript_text = ' '.join([snippet.text for snippet in fetched_transcript])
                        if transcript_text and len(transcript_text) > 50:
                            st.success(f" Found {transcript.language} transcript!")
                            return transcript_text, f"{transcript.language} transcript", transcript.language_code
                    except Exception as e:
                        continue
            except Exception as e:
                st.warning(f" Could not list transcripts: {e}")
                    
        except Exception as e:
            st.error(f" Transcript error: {str(e)}")
        
        st.error(" No transcript found for this video")
        return "", "No transcript available", "unknown"

# TTS Manager (unchanged)
class TTSManager:
    def text_to_speech(self, text: str) -> Optional[str]:
        try:
            if len(text) > 400:
                text = text[:397] + "..."
            tts = gTTS(text=text, lang='en', slow=False)
            output_file = "summary_audio.mp3"
            tts.save(output_file)
            return output_file
        except Exception as e:
            st.error(f" TTS Error: {e}")
            return None

# Main Application (unchanged)
class YouTubeSummaryApp:
    def __init__(self, api_key: str):
        self.youtube_manager = YouTubeAPIManager(api_key)
        self.ai_summarizer = AISummarizer()
        self.chatbot = AIChatBot()
        self.tts = TTSManager()

    def process_video(self, youtube_url: str, generate_audio: bool = False, summary_ratio: float = 0.3) -> Dict:
        """AI-powered processing pipeline"""
        start_time = time.time()
        
        try:
            # Extract video ID
            video_id = self.youtube_manager.extract_video_id(youtube_url)
            st.success(f" Video ID: {video_id}")
        except Exception as e:
            return {"error": f" URL Error: {e}"}

        # Get video details
        with st.spinner(" Fetching video details..."):
            video_details = self.youtube_manager.get_video_details(video_id)
            if not video_details:
                return {"error": " Could not fetch video details."}

        # Display video info
        st.subheader(" Video Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Title:** {video_details['title']}")
            st.write(f"**Channel:** {video_details['channel']}")
        with col2:
            st.write(f"**Views:** {video_details['views']}")
            st.write(f"**Likes:** {video_details['likes']}")

        # Get transcript using the new fetch() function
        with st.spinner(" Fetching transcript..."):
            transcript_text, source, language = self.youtube_manager.get_video_transcript(video_id)
            
            if not transcript_text or len(transcript_text) < 100:
                st.warning(" No transcript found. Using video description as fallback.")
                description = video_details.get('description', '')
                if len(description) > 100:
                    transcript_text = description
                    source = "Video description (fallback)"
                else:
                    return {"error": " No transcript or sufficient description available."}
            else:
                st.success(f" Successfully retrieved {source}")

        # Show transcript sample
        with st.expander(" View Transcript Sample", expanded=False):
            st.markdown('<div class="transcript-box">', unsafe_allow_html=True)
            st.text_area(
                "First 800 characters of transcript:",
                transcript_text[:800] + "..." if len(transcript_text) > 800 else transcript_text,
                height=200,
                key="transcript_sample"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.write(f"**Total transcript length:** {len(transcript_text):,} characters")

        # Load AI model and generate summary
        with st.spinner(" Loading model for summarization..."):
            self.ai_summarizer.load_model()
        
        with st.spinner(" Generating summary..."):
            if self.ai_summarizer.model_loaded:
                summary = self.ai_summarizer.summarize_with_ai(transcript_text, summary_ratio)
                summary_source = "AI Summary"
            else:
                summary = self.ai_summarizer._fallback_summarize(transcript_text, summary_ratio)
                summary_source = "Basic Summary (Model failed to load)"

        # Load chatbot model
        with st.spinner(" Loading chatbot..."):
            self.chatbot.load_model()

        # Generate audio if requested
        audio_file = None
        if generate_audio:
            with st.spinner(" Generating audio..."):
                audio_file = self.tts.text_to_speech(summary)

        total_time = time.time() - start_time
        
        return {
            "success": True,
            "video_id": video_id,
            "video_details": video_details,
            "content_source": source,
            "summary_source": summary_source,
            "detected_language": language,
            "original_content_length": len(transcript_text),
            "english_summary": summary,
            "full_transcript": transcript_text,
            "transcript_sample": transcript_text[:500],
            "audio_file": audio_file,
            "processing_time": total_time,
            "ai_model_used": self.ai_summarizer.model_loaded,
            "chatbot_loaded": self.chatbot.model_loaded
        }

# Enhanced Chat UI Component with better question suggestions
def render_chat_interface(full_transcript: str, summary: str, video_title: str):
    """Render the chat interface for Q&A about the video"""
    st.markdown("---")
    st.markdown(f"### 💬 Ask Questions About: *{video_title}*")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AIChatBot()
        st.session_state.chatbot.load_model()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(f"Ask about the video content..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_conversational_response(
                    question=prompt,
                    full_transcript=full_transcript,
                    summary=summary
                )
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("💡 Suggest Questions", use_container_width=True):
            suggest_questions(full_transcript, summary, video_title)

def suggest_questions(full_transcript: str, summary: str, video_title: str):
    """Suggest relevant questions to ask about the video"""
    # Extract key topics from summary and transcript
    summary_sentences = re.split(r'[.!?]+', summary)
    transcript_sentences = re.split(r'[.!?]+', full_transcript[:2000])  # First 2000 chars
    
    key_phrases = []
    
    # Extract potential key phrases (nouns and important verbs)
    for sentence in summary_sentences[:5] + transcript_sentences[:10]:
        sentence = sentence.strip()
        if len(sentence.split()) > 4 and len(sentence.split()) < 25:
            # Simple extraction of potential topics
            words = sentence.split()
            if len(words) > 6:
                # Take the core part of the sentence
                key_phrase = ' '.join(words[2:-2]) if len(words) > 8 else sentence
                key_phrases.append(key_phrase)
    
    # Remove duplicates
    key_phrases = list(dict.fromkeys(key_phrases))[:8]
    
    # Create question templates based on content type
    question_templates = []
    
    # Content-based questions
    if len(full_transcript) > 1000:
        question_templates.extend([
            "What is the main topic of this video?",
            "What are the key points discussed?",
            "Can you summarize the video in 3 points?",
            "What problem or issue does this video address?",
            "What solutions or recommendations are provided?",
            "Who is the target audience for this content?",
            "What are the most important takeaways?",
            "How does the speaker support their arguments?",
            "What examples or evidence are provided?",
            "What conclusions are reached at the end?"
        ])
    
    # Specific questions based on content analysis
    content_lower = (summary + " " + full_transcript[:1000]).lower()
    
    if any(word in content_lower for word in ['how to', 'tutorial', 'guide', 'steps']):
        question_templates.extend([
            "What are the steps mentioned in this tutorial?",
            "What tools or resources are needed?",
            "What are the common mistakes to avoid?"
        ])
    
    if any(word in content_lower for word in ['review', 'compare', 'versus', 'vs']):
        question_templates.extend([
            "What is being reviewed or compared?",
            "What are the pros and cons mentioned?",
            "What is the final recommendation?"
        ])
    
    if any(word in content_lower for word in ['explain', 'understand', 'concept', 'theory']):
        question_templates.extend([
            "Can you explain this concept in simple terms?",
            "What examples are used to explain this?",
            "Why is this concept important?"
        ])
    
    # Add key phrase specific questions
    for phrase in key_phrases[:3]:
        if len(phrase) > 10:
            question_templates.append(f"What does the video say about {phrase}?")
            question_templates.append(f"How is {phrase.split()[0]} important in this context?")
    
    # Select 6-8 diverse questions
    import random
    if len(question_templates) > 8:
        selected_questions = random.sample(question_templates, 8)
    else:
        selected_questions = question_templates
    
    st.markdown("**💡 Suggested questions you could ask:**")
    
    # Create columns for better layout
    cols = st.columns(2)
    for i, question in enumerate(selected_questions):
        with cols[i % 2]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                # Auto-ask the selected question
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.generate_conversational_response(
                        question=question,
                        full_transcript=full_transcript,
                        summary=summary
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

# Streamlit UI (unchanged)
def main():
    st.markdown('<h1 class="main-header"> Summarease </h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input("YouTube Data API Key", type="password", 
                               help="Get from https://console.cloud.google.com/")
        
        st.markdown("---")
        st.markdown("### 📝 Summary Settings")
        summary_ratio = st.slider(
            "Summary Length", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3,
            help="Higher values = more detailed summaries"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 Chatbot Settings")
        enable_chatbot = st.checkbox(
            "Enable AI Chatbot",
            value=True,
            help="Ask questions about the video content"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        **Summarease** combines:
        - YouTube transcript extraction
        - AI-powered summarization  
        - Intelligent Q&A chatbot
        - Audio summary generation
        """)
    
    # Main content
    st.markdown("### 📹 Enter YouTube Video URL")
    youtube_url = st.text_input(
        "YouTube URL", 
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Paste any YouTube video URL",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_audio = st.checkbox(
            "Generate Audio Summary", 
            value=False,
            help="Convert text summary to speech (adds a few seconds)"
        )
    with col2:
        process_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)
    
    if process_button:
        if not api_key:
            st.error("❌ Please enter your YouTube API key")
            return
        if not youtube_url:
            st.error("❌ Please enter a YouTube URL")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text(" Initializing...")
            progress_bar.progress(10)
            
            # Initialize app
            app = YouTubeSummaryApp(api_key)
            progress_bar.progress(30)
            
            status_text.text(" Processing video content...")
            result = app.process_video(youtube_url, generate_audio, summary_ratio)
            progress_bar.progress(80)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                return
                
            if not result.get("success", False):
                st.error(f"❌ {result.get('error', 'Unknown error occurred')}")
                return
            
            # Store results in session state for chatbot
            st.session_state.video_context = result.get('full_transcript', '')
            st.session_state.video_summary = result.get('english_summary', '')
            st.session_state.video_title = result.get('video_details', {}).get('title', 'Video')
            
            # Display results
            st.markdown("---")
            st.markdown("## ✅ Summary Generated!")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original", f"{result['original_content_length']:,} chars")
            with col2:
                st.metric("Summary", f"{len(result['english_summary']):,} chars")
            with col3:
                compression = ((result['original_content_length'] - len(result['english_summary'])) / result['original_content_length']) * 100
                st.metric("Compression", f"{compression:.1f}%")
            with col4:
                model_status = "🤖 AI" if result.get('ai_model_used', False) else "📄 Basic"
                st.metric("Model", model_status)
            
            # Summary
            st.markdown("### 📄 Summary")
            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
            st.write(result['english_summary'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Source information
            st.info(f"**Source:** {result['content_source']} | **Method:** {result['summary_source']} | **Time:** {result.get('processing_time', 0):.1f}s")
            
            # Audio
            if result['audio_file']:
                st.markdown("### 🔊 Audio Summary")
                st.audio(result['audio_file'])
            
            progress_bar.progress(100)
            status_text.text(f" Complete in {result.get('processing_time', 0):.1f} seconds!")
            
            # Show chatbot if enabled
            if enable_chatbot and result.get('chatbot_loaded', False):
                render_chat_interface(
                    result['full_transcript'],
                    result['english_summary'],
                    result['video_details']['title']
                )
            
        except Exception as e:
            st.error(f" An error occurred: {str(e)}")
            progress_bar.progress(0)
            status_text.text(" Failed")
    
    # Show chatbot if video was already processed
    elif 'video_context' in st.session_state and enable_chatbot:
        render_chat_interface(
            st.session_state.video_context,
            st.session_state.video_summary,
            st.session_state.video_title
        )

if __name__ == "__main__":
    main()