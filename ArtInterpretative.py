import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from openai import OpenAI
import plotly.express as px
import pandas as pd
import base64
import io

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

st.set_page_config(
    page_title="🎨 Art Interpretative",
    page_icon="🖌️",
    layout="wide"
)

# CSS para fondo rosita y encabezado
st.markdown(
    """
    <style>
    /* Fondo degradado rosa */
    .stApp {
        background: linear-gradient(135deg, #ffe6f0, #ffd6eb, #ffcce5, #ffb3da);
    }

    /* Encabezado */
    .custom-header {
        background-color: #FB9EC6; /* rosa intenso */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .custom-header h1 {
        color: white;
        font-size: 2.5em;
        font-family: 'Trebuchet MS', sans-serif;
        margin: 0;
    }
    </style>

    <div class="custom-header">
        <h1>🎨 Art Interpretative 🌸</h1>
    </div>
    """,
    unsafe_allow_html=True
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_models():
    model_style = tf.keras.models.load_model("StyleClass.keras", compile=False)
    model_category = tf.keras.models.load_model("CategoryClass.keras", compile=False)
    model_emotions = tf.keras.models.load_model("EmotionClass.keras", compile=False)
    return model_style, model_category, model_emotions

def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_palette(image, n_colors=7):
    img = image.resize((150, 150))
    img_array = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(img_array)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def display_palette(colors):
    cols = st.columns(len(colors))
    for i, color in enumerate(colors):
        cols[i].markdown(
            f"<div style='background-color: rgb{tuple(color)}; height: 50px; border-radius: 5px;'></div>",
            unsafe_allow_html=True
        )

def generate_interpretation(emotions, style, category, keywords, description=""):
    if client is None:
        return " No se encontró API Key de OpenAI. Muestra de interpretación local."

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
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generando interpretación: {e}"

# Cargar modelos
model_style, model_category, model_emotions = load_models()

st.markdown("💡 Upload your picture")
col_left, col_right = st.columns([1, 1])

uploaded_file = st.file_uploader("📂 Upload your picture", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Image uploaded", use_container_width=False, width=500)

    with col_results:
        img_array = preprocess_image(image)
        pred_style = model_style.predict(img_array)
        pred_category = model_category.predict(img_array)
        pred_emotions = model_emotions.predict(img_array)
        pred_style = pred_style[0]
        pred_category = pred_category[0]
        pred_emotions = pred_emotions[0]

        styles = ['Contemporary Art', 'Modern Art', 'Post Renaissance Art','Renaissance Art']
        categories = ['Abstract Art', 'Abstract Expressionism', 'Art Informel', 'Baroque', 
        'Color Field Painting', 'Cubism', 'Early Renaissance', 'Expressionism', 'High Renaissance',
        'Impressionism', 'Lyrical Abstraction', 'Magic Realism', 'Minimalism', 'Neo-Expressionism', 
        'Neoclassicism', 'Northern Renaissance', 'Pop Art', 'Post-Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Surrealism']
        emotions_list = ['agreeableness', 'anger','anticipation', 'arrogance','disagreeableness', 
            'disgust', 'fear','gratitude', 'happiness', 'humility','love', 'optimism', 
            'pessimism','regret', 'sadness', 'shame','shyness', 'surprise']

        style_result = styles[np.argmax(pred_style)]
        category_result = categories[np.argmax(pred_category)]
        emotion_result = emotions_list[np.argmax(pred_emotions)]

        # Paleta de colores
        colors = get_palette(image)
        st.subheader("🎨 Paleta de colores")
        display_palette(colors)

        # Keywords
        keywords = [style_result, category_result, emotion_result]

        with st.expander("🖌 Interpretación metadata"):
            interpretation = generate_interpretation(
                emotion_result, style_result, category_result, keywords
            )
            st.write(interpretation)

        # -------------------
        # Dashboard con tabs
        # -------------------
        tab1, tab2, tab3 = st.tabs(["🎨 Style", "🖼️ Category", "💖 Emotion"])

        with tab1:

            df_styles = pd.DataFrame({"Estilo": styles, "Probabilidad": pred_style})  # <--- corregido
            fig_styles = px.bar(df_styles, x="Estilo", y="Probabilidad",
                    title="Distribución de estilos",
                    color="Probabilidad", color_continuous_scale="Blues", width=1000, height=500)
            st.plotly_chart(fig_styles, use_container_width=True)

        with tab2:
            min_len_cat = min(len(categories), len(pred_category))
            categories = categories[:min_len_cat]
            pred_category = pred_category[:min_len_cat]
            df_categories = pd.DataFrame({"Categoría": categories, "Probabilidad": pred_category})  # <--- corregido
            fig_categories = px.bar(df_categories, x="Categoría", y="Probabilidad",
                        title="Distribución de categorías",
                        color="Probabilidad", color_continuous_scale="Purples",
                         width=1000, height=500)
            st.plotly_chart(fig_categories, use_container_width=True)

        with tab3:
            min_len_emo = min(len(emotions_list), len(pred_emotions))
            emotions_list = emotions_list[:min_len_emo]
            pred_emotions = pred_emotions[:min_len_emo]
            df_emotions = pd.DataFrame({"Emoción": emotions_list, "Probabilidad": pred_emotions})  # <--- corregido
            fig_emotions = px.bar(df_emotions, x="Emoción", y="Probabilidad",
                      title="Distribución de emociones",
                      color="Probabilidad", color_continuous_scale="Reds",
                      width=1000, height=500)
            st.plotly_chart(fig_emotions, use_container_width=True)

        # Resultados finales
        st.subheader("Results")
        st.markdown(f"**Style:** {style_result}")
        st.markdown(f"**Category:** {category_result}")
        st.markdown(f"**Emotion:** {emotion_result}")
        st.markdown(f"**Keyword:** {', '.join(keywords)}")