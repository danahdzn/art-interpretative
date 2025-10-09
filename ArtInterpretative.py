import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from openai import OpenAI
import plotly.express as px
import pandas as pd
import base64
from math import sqrt
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import hashlib
from googletrans import Translator

# translation
translator = Translator()
def translate_text(text):
    try:
        translated = translator.translate(text, src='en', dest='es').text
        return translated.capitalize()
    except Exception:
        return text.capitalize()

# -----------------------------
# Configuración OpenAI
# -----------------------------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

# -----------------------------
# Streamlit config y CSS
# -----------------------------
st.set_page_config(page_title="🎨 Art Interpretative", page_icon="🖌️", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #ffe6f0, #ffd6eb, #ffcce5, #ffb3da);}
.custom-header {
    background-color: #FB9EC6;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);}
.custom-header h1 {
    color: white;
    font-size: 2.5em;
    font-family: 'Trebuchet MS', sans-serif;
    margin: 0;}
</style>
<div class="custom-header">
    <h1>🎨 Art Interpretative 🌸</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Cargar modelos Hugging Face (solo 1 vez)
# -----------------------------
@st.cache_resource
def load_models_transformers():
    caption_model_name = "Salesforce/blip-image-captioning-large"
    caption_processor = BlipProcessor.from_pretrained(caption_model_name)
    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)

    emotion_model_name = "deepakshirkem/image-description_to_emotion"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForCausalLM.from_pretrained(emotion_model_name)

    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        framework="pt"
    )
    return caption_processor, caption_model, emotion_tokenizer, emotion_model, emotion_classifier

caption_processor, caption_model, emotion_tokenizer, emotion_model, emotion_classifier = load_models_transformers()

# -----------------------------
# Modelos Keras
# -----------------------------
@st.cache_resource
def load_models_keras():
    model_style = tf.keras.models.load_model("StyleClass.h5", compile=False)
    model_category = tf.keras.models.load_model("CategoryClass.h5", compile=False)
    return model_style, model_category

model_style, model_category = load_models_keras()

# -----------------------------
# Funciones auxiliares
# -----------------------------
emotion_colors = {
    "amor": (255, 105, 180), "felicidad": (255, 223, 0), "odio": (128, 0, 0),
    "enojo": (255, 0, 0), "deseo": (255, 0, 127), "tristeza": (30, 144, 255),
    "decepción": (112, 128, 144), "miedo": (75, 0, 130), "paz": (144, 238, 144)
}

def assign_emotion_to_color(color_rgb):
    min_distance = float('inf')
    closest_emotion = None
    for emotion, base_rgb in emotion_colors.items():
        distance = sqrt(sum((color_rgb[i] - base_rgb[i])**2 for i in range(3)))
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion
    return closest_emotion

def preprocess_image(image, target_size=(128,128)):
    img = image.resize(target_size)
    img_array = np.array(img)/255.0
    return np.expand_dims(img_array, axis=0)

def get_palette(image, n_colors=7):
    img = image.resize((150,150))
    img_array = np.array(img).reshape(-1,3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_array)
    return kmeans.cluster_centers_.astype(int)

def display_palette(colors):
    for color in colors:
        emotion = assign_emotion_to_color(color)
        st.markdown(f"""
        <div style='display:flex; align-items:center; margin-bottom:5px;'>
            <div style='background-color: rgb{tuple(color)}; width:50px; height:50px; border-radius:5px; margin-right:10px;'></div>
            <div style='font-weight:bold; color:#333;'>{emotion.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)

def generate_interpretation(emotions, style, category, keywords, description=""):
    if client is None: return "No se encontró API Key de OpenAI."
    prompt = f"""
Analiza esta pintura. 
- Estilo: {style}
- Categoría: {category}
- Emociones detectadas: {emotions}
- Palabras clave: {', '.join(keywords)}

{description}

Por favor escribe un breve resumen de la categoria y estilo al que pertenece la obra
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generando interpretación: {e}"

# -----------------------------
# Cachear predicciones por imagen
# -----------------------------
def hash_image(img):
    return hashlib.md5(img.tobytes()).hexdigest()

@st.cache_data
def predict_all(image):
    img_array = preprocess_image(image)
    pred_style = model_style.predict(img_array)[0]
    pred_category = model_category.predict(img_array)[0]

    styles = ['Contemporary Art', 'Modern Art', 'Post Renaissance Art','Renaissance Art']
    categories = ['Abstract Art', 'Abstract Expressionism', 'Art Informel', 'Baroque', 
                  'Color Field Painting', 'Cubism', 'Early Renaissance', 'Expressionism', 'High Renaissance',
                  'Impressionism', 'Lyrical Abstraction', 'Magic Realism', 'Minimalism', 'Neo-Expressionism', 
                  'Neoclassicism', 'Northern Renaissance', 'Pop Art', 'Post-Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Surrealism']

    style_result = styles[np.argmax(pred_style)]
    category_result = categories[np.argmax(pred_category)]

    # BLIP descripción
    inputs_blip = caption_processor(image.resize((224,224)), return_tensors="pt")
    caption_ids = caption_model.generate(**inputs_blip, max_new_tokens=50)
    description = caption_processor.decode(caption_ids[0], skip_special_tokens=True)

    # Emoción
    prompt_em = f"Describe the emotion conveyed by this image: {description}"
    inputs_em = emotion_tokenizer(prompt_em, return_tensors="pt")
    outputs_em = emotion_model.generate(**inputs_em, max_new_tokens=40)
    emotion_text = emotion_tokenizer.decode(outputs_em[0], skip_special_tokens=True)

    # Clasificador emociones
    emotions_scores = emotion_classifier(emotion_text)[0]
    emotions_list = [e['label'] for e in emotions_scores]
    pred_emotions = [e['score'] for e in emotions_scores]
    emotion_result = emotions_list[pred_emotions.index(max(pred_emotions))]

    # Paleta
    colors = get_palette(image)

    return style_result, category_result, emotion_result, description, emotion_text, colors, pred_style, pred_category, emotions_list, pred_emotions

# -----------------------------
# Interfaz principal
# -----------------------------
st.markdown("💡 Upload your picture")
uploaded_file = st.file_uploader("📂 Upload your picture", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_results = st.columns([1,1])
    with col_img: st.image(image, caption="Image uploaded", use_container_width=False, width=500)
    style_result, category_result, emotion_result, description, emotion_text, colors, pred_style, pred_category, emotions_list, pred_emotions = predict_all(image)

    with col_results:
        description_es = translate_text(description)
        emotion_text_es = translate_text(emotion_text)
        st.subheader("📝 Descripción de la obra")
        st.write(description_es)
        st.subheader("💭 Descripción emocional")
        st.write(emotion_text_es)

        st.subheader("🎨 Paleta de colores")
        display_palette(colors)

        # Keywords
        keywords = [style_result, category_result, emotion_result]
        with st.expander("🖌 Interpretación metadata"):
            interpretation = generate_interpretation(emotion_result, style_result, category_result, keywords, description)
            st.write(interpretation)

        pastel_styles = ['#FDCFE8', '#FADADD', '#FEE3E0', '#FFE0E6'] 
        pastel_categories = ['#D6CDEA', '#E8D6F2', '#F2E0F8', '#F8EAFB']
        pastel_emotions = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF']

        # Tabs dashboard
        tab1, tab2, tab3 = st.tabs(["🎨 Style", "🖼️ Category", "💖 Emotion"])
        with tab1:
            df_styles = pd.DataFrame({"Estilo": ['Contemporary Art', 'Modern Art', 'Post Renaissance Art','Renaissance Art'], "Probabilidad": pred_style})
            fig_styles = px.bar(df_styles, x="Estilo", y="Probabilidad",
                    title="Distribución de estilos",
                    color="Probabilidad", color_discrete_sequence=pastel_styles,
                    width=1000, height=500)
            st.plotly_chart(fig_styles, use_container_width=True)
        with tab2:
            df_categories = pd.DataFrame({"Categoría": ['Abstract Art', 'Abstract Expressionism', 'Art Informel', 'Baroque', 
                                                        'Color Field Painting', 'Cubism', 'Early Renaissance', 'Expressionism', 'High Renaissance',
                                                        'Impressionism', 'Lyrical Abstraction', 'Magic Realism', 'Minimalism', 'Neo-Expressionism', 
                                                        'Neoclassicism', 'Northern Renaissance', 'Pop Art', 'Post-Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Surrealism'], 
                                          "Probabilidad": pred_category})
            fig_categories = px.bar(df_categories, x="Categoría", y="Probabilidad",
                        title="Distribución de categorías",
                        color="Probabilidad", color_discrete_sequence=pastel_categories,
                        width=1000, height=500)
            st.plotly_chart(fig_categories, use_container_width=True)
        with tab3:
            df_emotions = pd.DataFrame({"Emoción": emotions_list, "Probabilidad": pred_emotions})
            fig_emotions = px.bar(df_emotions, x="Emoción", y="Probabilidad",
                      title="Distribución de emociones",
                      color="Probabilidad", color_discrete_sequence=pastel_emotions,
                      width=1000, height=500)
            st.plotly_chart(fig_emotions, use_container_width=True)

        st.subheader("Results")
        st.markdown(f"**Style:** {style_result}")
        st.markdown(f"**Category:** {category_result}")
        st.markdown(f"**Emotion:** {emotion_result}")