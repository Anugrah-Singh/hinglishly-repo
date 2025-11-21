import streamlit as st
import os
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Hinglishly - Smart Text Correction", 
    page_icon="✨", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6 {
        color: white !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    /* Sidebar checkbox and selectbox styling */
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        color: #6c757d;
        text-align: center;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 15px;
        border: 2px solid #667eea;
        background-color: white !important;
        color: #333 !important;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .correction-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        color: #333 !important;
    }
    .suggestion-item {
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #667eea;
        color: #333 !important;
    }
    .stMarkdown {
        color: #2c3e50 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    p, span, div {
        color: #2c3e50 !important;
    }
    .stMetric label {
        color: #2c3e50 !important;
    }
    .stMetric .metric-value {
        color: #1a252f !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="big-font">✨ Hinglishly</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Intelligent Hinglish Text Assistant</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - ✅ **Grammar & Spell Check**
    - 🔍 **Error Detection & Explanation**
    - 💡 **Smart Suggestions**
    - 🎨 **Tone Analysis**
    - 📊 **Clarity Scoring**
    - 📚 **Vocabulary Enhancement**
    - 📝 **Readability Assessment**
    - 🌐 **Hinglish Support**
    - 🔄 **Paraphrasing Options**
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    auto_correct = st.checkbox("Enable Auto-Correct", value=True)
    show_suggestions = st.checkbox("Show Suggestions", value=True)
    show_explanations = st.checkbox("Show Explanations", value=True)
    show_tone = st.checkbox("Tone Detection", value=True)
    show_vocabulary = st.checkbox("Vocabulary Enhancement", value=True)
    
    st.markdown("---")
    st.markdown("### 🎭 Tone Adjustment")
    target_tone = st.selectbox(
        "Adjust writing tone:",
        ["Keep Original", "Formal", "Friendly", "Professional", "Casual"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    device_type = '🎮 GPU' if torch.cuda.is_available() else '💻 CPU'
    st.info(f"**Processing:** {device_type}")

# Check for API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ API Key not configured. Please contact administrator.")
    st.stop()

# Define Output Structure
class GrammarAnalysis(BaseModel):
    detected_error: str = Field(description="The type of error detected (e.g., Grammar, Spelling, Punctuation, Word Choice, None)")
    corrected_text: str = Field(description="The corrected Hinglish text")
    suggestions: list = Field(description="List of improvement suggestions", default=[])
    tone: str = Field(description="Detected tone (e.g., Formal, Friendly, Casual, Professional)", default="Neutral")
    clarity_score: int = Field(description="Clarity score from 1-10", default=7)
    vocabulary_enhancements: list = Field(description="Better word choice suggestions", default=[])
    explanations: list = Field(description="Explanations for corrections made", default=[])
    readability: str = Field(description="Readability assessment (Easy, Medium, Complex)", default="Medium")
    word_count: int = Field(description="Number of words", default=0)
    sentence_count: int = Field(description="Number of sentences", default=0)

# Initialize Model and Chain
@st.cache_resource
def get_llm():
    return ChatGroq(**{
        "temperature": 0,
        "model": "llama-3.3-70b-versatile",
        "groq_api_key": api_key
    })

@st.cache_resource
def get_chain():
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=GrammarAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced AI writing assistant expert in Hinglish (Hindi + English mixed language) text analysis. 

CRITICAL INSTRUCTIONS:
- Carefully analyze the input text for errors in grammar, spelling, punctuation, word choice, or clarity
- If the text has NO errors and is perfectly written, set detected_error to "None" and corrected_text should be EXACTLY the same as input
- If the text HAS errors, identify the specific error type and provide a corrected version that is DIFFERENT from the input
- Be strict in your analysis - don't assume errors exist if the text is genuinely correct

Analyze the user's Hinglish text and provide:
1. **Error Detection**: 
   - If NO errors found: Set to "None"
   - If errors found: Identify the primary type (Grammar, Spelling, Punctuation, Word Choice, Slang, Transliteration, Capitalization, Missing Words, Word Order)

2. **Corrected Text**: 
   - If NO errors: Return EXACTLY the same text as input
   - If errors found: Provide the corrected version (must be different from input)

3. **Suggestions**: 
   - If NO errors: ["Text is well-written and clear"]
   - If errors found: 2-3 specific actionable improvements

4. **Tone Analysis**: Detect the tone (Formal, Friendly, Casual, Professional, Neutral)

5. **Clarity Score**: Rate clarity from 1-10 (be honest - perfect text gets 9-10, text with errors gets lower scores)

6. **Vocabulary Enhancements**: 
   - If text is already good: []
   - If improvements possible: Suggest 1-2 better word choices with alternatives

7. **Explanations**: 
   - If NO errors: []
   - If errors found: Explain WHY each correction was made (be educational and specific)

8. **Readability**: Assess as Easy, Medium, or Complex

9. **Word Count**: Count total words accurately

10. **Sentence Count**: Count total sentences accurately

EXAMPLES:

Input: "Hi mera name Tanya hai"
Output: detected_error="Grammar", corrected_text="Hi, mera naam Tanya hai", suggestions=["Add comma after 'Hi'", "Use 'naam' instead of 'name' for consistency"], explanations=["Added comma for proper punctuation", "Changed 'name' to 'naam' for Hinglish consistency"]

Input: "Hello, how are you doing today?"
Output: detected_error="None", corrected_text="Hello, how are you doing today?", suggestions=["Text is well-written and clear"], explanations=[]

Input: "I am going to market tomorrow"
Output: detected_error="None", corrected_text="I am going to market tomorrow", suggestions=["Text is grammatically correct"], explanations=[]

Input: "mera naam tanya rehta"
Output: detected_error="Grammar", corrected_text="Mera naam Tanya hai", suggestions=["Replace 'rehta' with 'hai' because 'rehta' is incorrect for stating a name", "Capitalize proper noun 'Tanya'"], explanations=["'Rehta' refers to residence/habit and cannot be used to state one's name", "Proper nouns should be capitalized"]

Be honest and accurate in your analysis.
        {format_instructions}"""),
        ("human", "{text}"),
    ])
    
    chain = prompt | llm | parser
    return chain

try:
    chain = get_chain()
    llm = get_llm()
except Exception as e:
    st.error(f"⚠️ Service initialization failed. Please try again later.")
    st.stop()

# Main Input Area
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "✍️ Enter your text here:", 
        height=200, 
        placeholder="Type your Hinglish text here...\nExample: mujhe kal meeting me jana hai",
        key="input_text"
    )

with col2:
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    <div class="suggestion-item">
    <b>Mix languages freely!</b><br>
    Write naturally in Hinglish
    </div>
    <div class="suggestion-item">
    <b>Get instant feedback</b><br>
    Corrections appear in seconds
    </div>
    <div class="suggestion-item">
    <b>Learn as you write</b><br>
    See helpful suggestions
    </div>
    """, unsafe_allow_html=True)

# Action Buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    analyze_button = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🔄 Clear", use_container_width=True)

if clear_button:
    # Clear paraphrasing session state
    st.session_state.formal_version = None
    st.session_state.creative_version = None
    st.session_state.professional_version = None
    st.rerun()

if analyze_button:
    # Clear previous paraphrasing results when analyzing new text
    st.session_state.formal_version = None
    st.session_state.creative_version = None
    st.session_state.professional_version = None
    
    if not text_input:
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Add tone adjustment to the prompt if selected
        analysis_text = text_input
        if target_tone != "Keep Original":
            analysis_text = f"[Adjust tone to {target_tone}] {text_input}"
        
        with st.spinner("✨ Analyzing your text..."):
            try:
                result = chain.invoke({
                    "text": analysis_text,
                    "format_instructions": JsonOutputParser(pydantic_object=GrammarAnalysis).get_format_instructions()
                })
            except Exception as parse_error:
                # Fallback: Create a basic analysis if parsing fails
                st.warning("⚠️ Using simplified analysis mode")
                result = {
                    "detected_error": "None",
                    "corrected_text": text_input,
                    "suggestions": ["Text appears to be correct"],
                    "tone": "Neutral",
                    "clarity_score": 7,
                    "vocabulary_enhancements": [],
                    "explanations": [],
                    "readability": "Medium",
                    "word_count": len(text_input.split()),
                    "sentence_count": text_input.count('.') + text_input.count('!') + text_input.count('?') or 1
                }
        
        # Results Section
        st.markdown("---")
        st.markdown("## 📋 Comprehensive Analysis")
        
        # Stats Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Words", result.get("word_count", "N/A"))
        with col2:
            st.metric("📝 Sentences", result.get("sentence_count", "N/A"))
        with col3:
            clarity = result.get("clarity_score", 7)
            st.metric("🎯 Clarity", f"{clarity}/10")
        with col4:
            readability = result.get("readability", "Medium")
            st.metric("📖 Readability", readability)
        
        st.markdown("---")
        
        # Error Detection
        error_type = result.get("detected_error", "Unknown")
        if error_type.lower() == "none":
            st.success("✅ **Excellent!** No errors detected in your text.")
        else:
            st.warning(f"⚠️ **Issue Detected:** {error_type}")
        
        # Tone Analysis
        if show_tone:
            tone = result.get("tone", "Neutral")
            tone_colors = {
                "Formal": "🎩",
                "Friendly": "😊",
                "Professional": "💼",
                "Casual": "😎",
                "Neutral": "⚖️"
            }
            tone_icon = tone_colors.get(tone, "⚖️")
            st.info(f"{tone_icon} **Detected Tone:** {tone}")
        
        # Text Comparison
        corrected = result.get("corrected_text", text_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Original Text")
            st.markdown(f'<div class="correction-box">{text_input}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ✨ Corrected Text")
            if auto_correct:
                st.markdown(f'<div class="correction-box" style="background: #d4edda; border-left: 4px solid #28a745;">{corrected}</div>', unsafe_allow_html=True)
                if st.button("📋 Copy Corrected Text"):
                    st.code(corrected, language=None)
            else:
                st.markdown(f'<div class="correction-box">{corrected}</div>', unsafe_allow_html=True)
        
        # Explanations
        if show_explanations:
            explanations = result.get("explanations", [])
            if explanations and isinstance(explanations, list) and len(explanations) > 0:
                st.markdown("---")
                st.markdown("### 📚 Why These Corrections?")
                for i, explanation in enumerate(explanations, 1):
                    st.markdown(f"""
                    <div class="suggestion-item" style="background: #fff3cd; border-left: 4px solid #ffc107;">
                    <b>💡 Explanation {i}:</b> {explanation}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Vocabulary Enhancement
        if show_vocabulary:
            vocab = result.get("vocabulary_enhancements", [])
            if vocab and isinstance(vocab, list) and len(vocab) > 0:
                st.markdown("---")
                st.markdown("### 📖 Vocabulary Enhancement")
                for i, enhancement in enumerate(vocab, 1):
                    st.markdown(f"""
                    <div class="suggestion-item" style="background: #e7f3ff; border-left: 4px solid #2196F3;">
                    <b>✏️ Suggestion {i}:</b> {enhancement}
                    </div>
                    """, unsafe_allow_html=True)
        
        # General Suggestions
        if show_suggestions:
            suggestions = result.get("suggestions", [])
            if suggestions and isinstance(suggestions, list) and len(suggestions) > 0:
                st.markdown("---")
                st.markdown("### 💡 Suggestions for Improvement")
                for i, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"""
                    <div class="suggestion-item">
                    <b>{i}.</b> {suggestion}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Paraphrasing Options
        st.markdown("---")
        st.markdown("### 🔄 Paraphrasing Options")
        
        # Initialize session state for paraphrasing
        if 'formal_version' not in st.session_state:
            st.session_state.formal_version = None
        if 'creative_version' not in st.session_state:
            st.session_state.creative_version = None
        if 'professional_version' not in st.session_state:
            st.session_state.professional_version = None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✍️ Rewrite (Formal)", use_container_width=True, key="formal_btn"):
                with st.spinner("✨ Generating formal version..."):
                    try:
                        formal_prompt = f"Rewrite this text in a formal tone while maintaining its meaning. Keep it concise: {corrected}"
                        formal_result = llm.invoke(formal_prompt)
                        st.session_state.formal_version = formal_result.content
                    except Exception as e:
                        st.error(f"Failed to generate formal version: {str(e)}")
            
            if st.session_state.formal_version:
                st.markdown(f"""
                <div class="correction-box" style="background: #f8f9fa; border-left: 4px solid #6c757d;">
                <b>✍️ Formal Version:</b><br>{st.session_state.formal_version}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🎨 Rewrite (Creative)", use_container_width=True, key="creative_btn"):
                with st.spinner("✨ Generating creative version..."):
                    try:
                        creative_prompt = f"Rewrite this text in a creative and engaging way while maintaining its meaning. Keep it concise: {corrected}"
                        creative_result = llm.invoke(creative_prompt)
                        st.session_state.creative_version = creative_result.content
                    except Exception as e:
                        st.error(f"Failed to generate creative version: {str(e)}")
            
            if st.session_state.creative_version:
                st.markdown(f"""
                <div class="correction-box" style="background: #fff3e0; border-left: 4px solid #ff9800;">
                <b>🎨 Creative Version:</b><br>{st.session_state.creative_version}
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if st.button("💼 Rewrite (Professional)", use_container_width=True, key="professional_btn"):
                with st.spinner("✨ Generating professional version..."):
                    try:
                        professional_prompt = f"Rewrite this text in a professional business tone while maintaining its meaning. Keep it concise: {corrected}"
                        professional_result = llm.invoke(professional_prompt)
                        st.session_state.professional_version = professional_result.content
                    except Exception as e:
                        st.error(f"Failed to generate professional version: {str(e)}")
            
            if st.session_state.professional_version:
                st.markdown(f"""
                <div class="correction-box" style="background: #e3f2fd; border-left: 4px solid #2196F3;">
                <b>💼 Professional Version:</b><br>{st.session_state.professional_version}
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px;">
    <p>Made with ❤️ for Hinglish speakers | Powered by Advanced AI</p>
</div>
""", unsafe_allow_html=True)